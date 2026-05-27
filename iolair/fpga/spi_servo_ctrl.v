// ============================================================
//  spi_servo_ctrl.v
//  Tang Nano 20K (GW2AR-18C) @ 27 MHz
//  Dual Servo PWM Controller — SPI + DIP switch fallback
//
//  Modo de operación (pin spi_enable):
//    spi_enable = 0  →  DIP switch  (pb1, pb2)  igual que tu .v original
//    spi_enable = 1  →  SPI desde Jetson Nano    256 pasos continuos
//
//  Protocolo SPI (MODE 0, MSB first, 500 kHz):
//    Trama: [0xAB][servo1_byte][servo2_byte]
//    0x00 = reversa máx  |  0x7F = stop  |  0xFF = adelante máx
//
//  PWM estándar servo: 50 Hz (periodo 20 ms)
//    1.0 ms → reversa máxima  (byte 0x00)
//    1.5 ms → stop             (byte 0x7F)
//    2.0 ms → adelante máximo  (byte 0xFF)
//
//  Tabla DIP (compatibilidad con servo_control.v original):
//    pb1=0, pb2=1  →  1.0 ms  (reversa)
//    pb1=0, pb2=0  →  1.5 ms  (stop / neutro)
//    pb1=1, pb2=0  →  2.0 ms  (adelante)
// ============================================================

module spi_servo_ctrl (
    // ── Sistema ──────────────────────────────────────────────
    input  wire clk,         // 27 MHz  (IO 4)
    // ── DIP switch fallback (activo alto) ────────────────────
    input  wire pb1,         // IO 25
    input  wire pb2,         // IO 26
    // ── Selector de modo ─────────────────────────────────────
    input  wire spi_enable,  // IO libre (ver .cst) — 0=DIP, 1=SPI
    // ── SPI slave (MODE 0: CPOL=0 CPHA=0) ───────────────────
    input  wire sck,         // IO libre — Jetson Pin 23
    input  wire mosi,        // IO libre — Jetson Pin 19
    output wire miso,        // IO libre — Jetson Pin 21  (tied low)
    input  wire cs_n,        // IO libre — Jetson Pin 24
    // ── Salidas PWM ──────────────────────────────────────────
    output wire pwm1,        // IO 27 — señal naranja Servo 1
    output wire pwm2         // IO 28 — señal naranja Servo 2
);

// ============================================================
// 1.  Parámetros PWM  (idénticos a tu servo_control.v original)
// ============================================================
localparam [19:0] PWM_PERIOD   = 20'd540_000;  // 20 ms @ 27 MHz
localparam [19:0] PULSE_MIN    = 20'd27_000;   //  1.0 ms
localparam [19:0] PULSE_NEUTRAL= 20'd40_500;   //  1.5 ms
localparam [19:0] PULSE_MAX    = 20'd54_000;   //  2.0 ms
// Paso SPI: (54 000 - 27 000) / 255 ≈ 106 ciclos/LSB
localparam [19:0] PULSE_STEP   = 20'd106;

// ============================================================
// 2.  Contador de periodo PWM  (mismo que tu versión original)
// ============================================================
reg [19:0] pwm_cnt;

always @(posedge clk) begin
    if (pwm_cnt >= PWM_PERIOD - 20'd1)
        pwm_cnt <= 20'd0;
    else
        pwm_cnt <= pwm_cnt + 20'd1;
end

// ============================================================
// 3.  Sincronizador SPI → dominio 27 MHz  (3 etapas)
//     Necesario porque SCK/MOSI/CS_N vienen de un reloj externo
// ============================================================
reg [2:0] sck_s, cs_s, mosi_s;

always @(posedge clk) begin
    sck_s  <= {sck_s[1:0],  sck};
    cs_s   <= {cs_s[1:0],   cs_n};
    mosi_s <= {mosi_s[1:0], mosi};
end

wire sck_r    = sck_s[2];
wire cs_r     = cs_s[2];
wire mosi_r   = mosi_s[2];
wire sck_rise = (sck_s[2:1] == 2'b01);  // flanco subida SCK
wire cs_fall  = (cs_s[2:1]  == 2'b10);  // CS activa (↓)
wire cs_rise  = (cs_s[2:1]  == 2'b01);  // CS inactiva (↑) = abort

// ============================================================
// 4.  Receptor SPI  (3 bytes por trama)
// ============================================================
reg [7:0] shift_reg;
reg [2:0] bit_cnt;
reg [1:0] byte_cnt;
reg [7:0] rx_buf [0:1];   // byte 0 (header) y byte 1 (servo1)

reg [7:0] servo1_reg;
reg [7:0] servo2_reg;

always @(posedge clk) begin
    // Reset de estado: inicio de trama o abort por CS↑
    if (cs_fall || cs_rise) begin
        bit_cnt  <= 3'd0;
        byte_cnt <= 2'd0;
    end

    // Muestrear MOSI en cada flanco ↑ de SCK mientras CS activo
    else if (!cs_r && sck_rise) begin
        shift_reg <= {shift_reg[6:0], mosi_r};
        bit_cnt   <= bit_cnt + 3'd1;   // 3 bits → overflow 7→0

        if (&bit_cnt) begin  // bit_cnt == 7 → byte completo
            case (byte_cnt)
                2'd0: begin
                    rx_buf[0] <= {shift_reg[6:0], mosi_r};  // header
                    byte_cnt  <= 2'd1;
                end
                2'd1: begin
                    rx_buf[1] <= {shift_reg[6:0], mosi_r};  // servo1
                    byte_cnt  <= 2'd2;
                end
                2'd2: begin
                    // Paquete completo: validar header antes de mover servos
                    if (rx_buf[0] == 8'hAB) begin
                        servo1_reg <= rx_buf[1];
                        // byte 2 (servo2): leer valor combinacional directo
                        // (non-blocking aún no committó rx_buf[2])
                        servo2_reg <= {shift_reg[6:0], mosi_r};
                    end
                    byte_cnt <= 2'd0;
                end
                default: byte_cnt <= 2'd0;
            endcase
        end
    end
end

assign miso = 1'b0;  // sin readback en esta versión

// ============================================================
// 5.  Decodificador DIP  (idéntico a tu servo_control.v original)
// ============================================================
reg [19:0] pw_dip;

always @(*) begin
    casex ({pb1, pb2})
        2'b01:   pw_dip = PULSE_MIN;     // reversa
        2'b10:   pw_dip = PULSE_MAX;     // adelante
        default: pw_dip = PULSE_NEUTRAL; // stop / neutro
    endcase
end

// ============================================================
// 6.  Cálculo de ancho de pulso SPI  (256 pasos continuos)
// ============================================================
wire [19:0] pw_spi1 = PULSE_MIN + ({12'd0, servo1_reg} * PULSE_STEP);
wire [19:0] pw_spi2 = PULSE_MIN + ({12'd0, servo2_reg} * PULSE_STEP);

// ============================================================
// 7.  Mux modo  +  salidas PWM
//     spi_enable=0 → ambos servos siguen DIP (control en paralelo)
//     spi_enable=1 → cada servo sigue su byte SPI individual
// ============================================================
wire [19:0] pulse1 = spi_enable ? pw_spi1 : pw_dip;
wire [19:0] pulse2 = spi_enable ? pw_spi2 : pw_dip;

assign pwm1 = (pwm_cnt < pulse1);
assign pwm2 = (pwm_cnt < pulse2);

endmodule