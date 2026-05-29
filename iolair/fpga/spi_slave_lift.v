// ============================================================
//  spi_slave_lift.v
//  Tang Nano 20K (GW2AR-18C) @ 27 MHz
//  Dual 360° Servo PWM via SPI MODE 0  —  MSB first
//
//  Protocolo (3 bytes):
//    [0xAB] [servo1_byte] [servo2_byte]
//     0x00  = reversa máx  (~1.0 ms)
//     0x7F  = stop         (~1.5 ms)
//     0xFF  = adelante máx (~2.0 ms)
//
//  Funciona con:
//    • CS único para los 3 bytes  (spidev.xfer2 normal)
//    • CS por byte                (quirk de algunos kernels Jetson)
// ============================================================
module spi_slave_lift (
    input  wire clk,        // 27 MHz           — pin 4
    input  wire sck,        // Jetson J41-23     — pin 72
    input  wire mosi,       // Jetson J41-19     — pin 71
    input  wire cs_n,       // Jetson J41-24     — pin 49  (activo bajo)
    output wire pwm1,       // Servo 1 naranja   — pin 27
    output wire pwm2,       // Servo 2 naranja   — pin 28
    output wire cs_debug    // Debug / osciloscopio — pin 15
);

// ============================================================
// 1.  PWM  50 Hz  —  rango 1.0 ms … 2.0 ms
//     27 MHz × 1.0 ms = 27 000 ciclos  (0x00)
//     27 MHz × 1.5 ms = 40 500 ciclos  (0x7F  → 127 × 106 + 27000 = 40 462 ≈ OK)
//     27 MHz × 2.0 ms = 54 000 ciclos  (0xFF  → 255 × 106 + 27000 = 54 030 ≈ OK)
// ============================================================
localparam [19:0] PWM_PERIOD = 20'd540_000;  // 20 ms
localparam [19:0] PULSE_MIN  = 20'd27_000;   //  1.0 ms
localparam [19:0] PULSE_STEP = 20'd106;      //  (54000-27000) / 255 ≈ 105.9

reg [19:0] pwm_cnt;
always @(posedge clk)
    pwm_cnt <= (pwm_cnt >= PWM_PERIOD - 20'd1) ? 20'd0 : pwm_cnt + 20'd1;

// ============================================================
// 2.  Sincronizador 3 etapas  (anti-metaestabilidad)
//     Pipeline:  [0] = más nuevo  …  [2] = más viejo
// ============================================================
reg [2:0] sck_s, cs_s, mosi_s;

always @(posedge clk) begin
    sck_s  <= {sck_s[1:0],  sck};
    cs_s   <= {cs_s[1:0],   cs_n};
    mosi_s <= {mosi_s[1:0], mosi};
end

wire cs_r     =  cs_s[2];              // CS sincronizado  (1 = inactivo)
// ── FIX: usar etapa intermedia [1] para MOSI ──────────────
// mosi_s[1] está muestreado en el mismo ciclo en que sck_s[1]
// ya muestra el flanco subida → mejor alineación temporal.
wire mosi_r   = mosi_s[1];
wire sck_rise = (sck_s[2:1] == 2'b01); // flanco subida SCK
wire cs_fall  = (cs_s[2:1]  == 2'b10); // CS baja  → inicio de trama / byte
wire cs_rise  = (cs_s[2:1]  == 2'b01); // CS sube  → fin   de trama / byte

// ============================================================
// 3.  Receptor SPI  —  shift register de 24 bits deslizante
//
//  ── Por qué NO se resetea shift_reg en cs_fall ─────────────
//  Si el driver de Jetson rompe CS entre cada byte (comportamiento
//  conocido de spidev en algunos kernels de Jetson Nano/TX2),
//  necesitamos acumular los 3 bytes a través de 3 afirmaciones
//  consecutivas de CS:
//
//    CS1 (0xAB)     → shift_reg = 0x00_00_AB
//    CS2 (servo1)   → shift_reg = 0x00_AB_s1
//    CS3 (servo2)   → shift_reg = 0xAB_s1_s2  ← validamos aquí ✓
//
//  Con CS único (xfer2 correcto) también funciona porque los
//  24 bits llegan de corrido y el resultado final es el mismo.
//
//  ── Validación al subir CS ─────────────────────────────────
//  bit_cnt cuenta los bits recibidos en ESTE ciclo de CS.
//  Al subir CS, bit_cnt == 0  significa que se recibieron
//  exactamente N × 8 bits (el contador es de 3 bits → wrap
//  automático cada 8 pulsos). Esto descarta tramas truncadas.
// ============================================================
reg [23:0] shift_reg;           // ventana deslizante de los últimos 24 bits
reg [ 2:0] bit_cnt;             // bits recibidos en el CS activo actual

reg [7:0] servo1_reg = 8'd127;  // stop al arrancar
reg [7:0] servo2_reg = 8'd127;

always @(posedge clk) begin

    // ── Reset de bit_cnt al inicio de cada ciclo CS ────────
    if (cs_fall)
        bit_cnt <= 3'd0;
    else if (!cs_r && sck_rise)
        bit_cnt <= bit_cnt + 3'd1;  // wrap automático 7 → 0 cada 8 bits

    // ── Desplazamiento: bit nuevo entra por el LSB ─────────
    if (!cs_r && sck_rise)
        shift_reg <= {shift_reg[22:0], mosi_r};

    // ── Validación al subir CS ─────────────────────────────
    // bit_cnt == 0  al subir CS  →  se recibió exactamente N×8 bits
    // shift_reg[23:16] == 0xAB   →  header válido en los últimos 24 bits
    if (cs_rise && (bit_cnt == 3'd0)) begin
        if (shift_reg[23:16] == 8'hAB) begin
            servo1_reg <= shift_reg[15:8];
            servo2_reg <= shift_reg[7:0];
        end
    end

end

// ============================================================
// 4.  Salidas PWM
// ============================================================
wire [19:0] pw1 = PULSE_MIN + ({12'd0, servo1_reg} * PULSE_STEP);
wire [19:0] pw2 = PULSE_MIN + ({12'd0, servo2_reg} * PULSE_STEP);

assign pwm1     = (pwm_cnt < pw1);
assign pwm2     = (pwm_cnt < pw2);
assign cs_debug = cs_r;           // 0 cuando CS activo → fácil de medir

endmodule