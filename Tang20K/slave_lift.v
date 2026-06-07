// ============================================================
//  spi_slave_lift.v  —  Tang Nano 20K (GW2AR-18C) @ 27 MHz
//  Dual 360° Servo PWM + State Machine + MISO feedback
//  v2 — fixes aplicados
//
//  ── Fix F1 (CRÍTICO): timer_fired flag ──────────────────────
//  La versión anterior tenía una race condition entre spi_new y
//  timer_done. Cuando la Jetson enviaba un CMD_PING exactamente
//  en el ciclo donde dur_cnt==1, el bloque "if (spi_new)" tenía
//  prioridad sobre "else if (timer_done)", el timer_done era
//  ignorado, dur_cnt caía a 0 en el decrementador, y el
//  siguiente ciclo timer_done=(0==1)=FALSE → SM atascada en
//  TO_N1/TO_N2/LIFTING/LOWERING para siempre.
//
//  Solución: registro timer_fired de 1 bit.
//    - Se activa (seteado) cuando dur_cnt llega a 1 Y sm_moving=1.
//    - Se limpia al ser consumido por la SM.
//    - La SM lo procesa EN EL SIGUIENTE CICLO, después de que el
//      bloque spi_new ya se resolvió.
//    - spi_new y timer_fired pueden ser TRUE en el mismo ciclo
//      sin conflicto: spi_new actualiza la SM en un ciclo,
//      timer_fired estará pendiente para el ciclo siguiente y
//      la SM actuará sobre el resultado del spi_new.
//    - Si spi_new fue un STOP (SM→IDLE, sm_moving→FALSE), el
//      flag timer_fired pendiente se descarta porque la rama
//      de la SM guarda la condición (sm_state != IDLE cuando
//      se procesa el flag).
//
//  ── Fix F2: bit_cnt robusto (5 bits) ───────────────────────
//  La versión anterior usaba bit_cnt de 3 bits para contar 24
//  bits. Funcionaba por coincidencia (24 mod 8 = 0), pero un
//  pulso extra de SCK por ruido dejaba bit_cnt ≠ 0 al final y
//  spi_new nunca se activaba → comando ignorado silenciosamente.
//  Solución: bit_cnt de 5 bits. spi_new se activa solo cuando
//  bit_cnt==23 (exactamente 24 bits recibidos).
//
//  ── Fix F3: mosi_r alineado con sck_rise ───────────────────
//  La versión anterior usaba mosi_s[1] (2 etapas de sync) y
//  sck_rise basado en sck_s[2:1] (3 etapas de sync). MOSI
//  llegaba 1 ciclo antes que SCK al lógico, pudiendo capturar
//  el bit previo en condiciones de skew adverso.
//  Solución: mosi_r = mosi_s[2] (misma etapa que sck).
//
//  ── Protocolo SPI (sin cambios) ────────────────────────────
//  TX [0xAC][cmd][0x00]  RX [sm_state_prev][0x00][0x00]
//  CMD_PING 0xFF = solo leer estado, sin cambiar nada
//
//  sm_state en MISO (byte 0 devuelto = estado ANTES del cmd):
//    0=IDLE  1=MAN_UP  2=MAN_DOWN
//    3=TO_N1  4=AT_N1  5=TO_N2  6=AT_N2
//    7=LIFTING  8=HOLD  9=LOWERING
// ============================================================
module spi_slave_lift (
    input  wire clk,
    input  wire sck,
    input  wire mosi,
    input  wire cs_n,
    output wire miso,       // Jetson J41-21  — pin 53
    output wire pwm1,       // Servo 1        — pin 27
    output wire pwm2,       // Servo 2        — pin 28
    output wire cs_debug    // Debug          — pin 15
);

// ============================================================
// 1.  PWM  50 Hz
// ============================================================
localparam [19:0] PWM_PERIOD = 20'd540_000;
localparam [19:0] PULSE_MIN  = 20'd27_000;
localparam [19:0] PULSE_STEP = 20'd106;

reg [19:0] pwm_cnt;
always @(posedge clk)
    pwm_cnt <= (pwm_cnt >= PWM_PERIOD - 20'd1) ? 20'd0 : pwm_cnt + 20'd1;

// ============================================================
// 2.  Sincronizador 3 etapas
// ============================================================
reg [2:0] sck_s, cs_s, mosi_s;
always @(posedge clk) begin
    sck_s  <= {sck_s[1:0],  sck};
    cs_s   <= {cs_s[1:0],   cs_n};
    mosi_s <= {mosi_s[1:0], mosi};
end

wire cs_r     = cs_s[2];
// Fix F3: mosi_r alineado a la misma etapa de sync que sck_rise
wire mosi_r   = mosi_s[2];
wire sck_rise = (sck_s[2:1] == 2'b01);
wire sck_fall = (sck_s[2:1] == 2'b10);
wire cs_fall  = (cs_s[2:1]  == 2'b10);
wire cs_rise  = (cs_s[2:1]  == 2'b01);

// ============================================================
// 3.  Receptor SPI (shift register deslizante)
//     Fix F2: bit_cnt de 5 bits, spi_new cuando bit_cnt==23
// ============================================================
reg [23:0] shift_reg;
// Fix F2: 5 bits para contar 0..23 sin overflow
reg [ 4:0] bit_cnt;

always @(posedge clk) begin
    if (cs_fall)
        bit_cnt <= 5'd0;
    else if (!cs_r && sck_rise)
        bit_cnt <= bit_cnt + 5'd1;

    if (!cs_r && sck_rise)
        shift_reg <= {shift_reg[22:0], mosi_r};
end

// Fix F2: spi_new solo cuando recibimos exactamente 24 bits
wire spi_new = cs_rise && (bit_cnt == 5'd24);

// ============================================================
// 4.  Parámetros SM
// ============================================================
localparam [3:0] SM_IDLE     = 4'd0;
localparam [3:0] SM_MAN_UP   = 4'd1;
localparam [3:0] SM_MAN_DOWN = 4'd2;
localparam [3:0] SM_TO_N1    = 4'd3;
localparam [3:0] SM_AT_N1    = 4'd4;
localparam [3:0] SM_TO_N2    = 4'd5;
localparam [3:0] SM_AT_N2    = 4'd6;
localparam [3:0] SM_LIFTING  = 4'd7;
localparam [3:0] SM_HOLD     = 4'd8;
localparam [3:0] SM_LOWERING = 4'd9;

localparam [7:0] CMD_PING     = 8'hFF;
localparam [7:0] CMD_STOP     = 8'h00;
localparam [7:0] CMD_MAN_UP   = 8'h20;
localparam [7:0] CMD_MAN_DOWN = 8'h21;
localparam [7:0] CMD_GO_N1    = 8'h10;
localparam [7:0] CMD_GO_N2    = 8'h11;
localparam [7:0] CMD_GO_HOLD  = 8'h12;
localparam [7:0] CMD_GO_DOWN  = 8'h13;

localparam [7:0] BYTE_UP   = 8'd165;
localparam [7:0] BYTE_DOWN = 8'd76;
localparam [7:0] BYTE_STOP = 8'd127;

localparam [25:0] DUR_N1      = 26'd12_400_000;
localparam [25:0] DUR_N2      = 26'd37_700_000;
localparam [25:0] DUR_N1_HOLD = 26'd20_000_000;
localparam [25:0] DUR_N2_HOLD = 26'd18_000_000;
localparam [25:0] DUR_DOWN    = 26'd55_700_000;
localparam [25:0] DUR_MAN     = 26'd56_700_000;

// ============================================================
// 5.  Registros SM
// ============================================================
reg [3:0]  sm_state   = SM_IDLE;
reg [25:0] dur_cnt    = 26'd0;
reg [7:0]  servo1_reg = BYTE_STOP;
reg [7:0]  servo2_reg = BYTE_STOP;

wire sm_moving = (sm_state == SM_MAN_UP)  || (sm_state == SM_MAN_DOWN) ||
                 (sm_state == SM_TO_N1)   || (sm_state == SM_TO_N2)    ||
                 (sm_state == SM_LIFTING) || (sm_state == SM_LOWERING);

// Fix F1: timer_fired flag — se activa cuando dur_cnt llega a 1
// mientras la SM está en movimiento. Se procesa en el ciclo
// siguiente, después de que cualquier spi_new ya fue atendido.
// Esto elimina la race condition spi_new vs timer_done.
reg timer_fired = 1'b0;

// ============================================================
// 6.  SM + decodificador SPI
// ============================================================
always @(posedge clk) begin

    // ── Decrementador del timer ─────────────────────────────
    // Corre independientemente; solo para cuando dur_cnt llega a 0.
    if (sm_moving && dur_cnt > 26'd0)
        dur_cnt <= dur_cnt - 26'd1;

    // Fix F1: detectar cuando dur_cnt está a punto de llegar a 0.
    // Registramos el flag cuando dur_cnt==1 (antes del último dec).
    // El flag se limpia al ser consumido por la SM en la rama siguiente.
    if (sm_moving && dur_cnt == 26'd1)
        timer_fired <= 1'b1;

    // ── Prioridad 1: nuevo frame SPI completo ────────────────
    if (spi_new) begin
        case (shift_reg[23:16])

            8'hAB: begin
                servo1_reg  <= shift_reg[15:8];
                servo2_reg  <= shift_reg[7:0];
                sm_state    <= SM_IDLE;
                dur_cnt     <= 26'd0;
                timer_fired <= 1'b0;   // abortar timer pendiente
            end

            8'hAC: begin
                case (shift_reg[15:8])
                    CMD_PING: ; // no-op: solo devuelve sm_state en MISO

                    CMD_STOP: begin
                        sm_state    <= SM_IDLE;
                        servo1_reg  <= BYTE_STOP;
                        dur_cnt     <= 26'd0;
                        timer_fired <= 1'b0;   // abortar timer pendiente
                    end
                    CMD_MAN_UP: begin
                        sm_state    <= SM_MAN_UP;
                        servo1_reg  <= BYTE_UP;
                        dur_cnt     <= DUR_MAN;
                        timer_fired <= 1'b0;
                    end
                    CMD_MAN_DOWN: begin
                        sm_state    <= SM_MAN_DOWN;
                        servo1_reg  <= BYTE_DOWN;
                        dur_cnt     <= DUR_MAN;
                        timer_fired <= 1'b0;
                    end
                    CMD_GO_N1: begin
                        if (sm_state == SM_IDLE) begin
                            sm_state   <= SM_TO_N1;
                            servo1_reg <= BYTE_UP;
                            dur_cnt    <= DUR_N1;
                        end
                    end
                    CMD_GO_N2: begin
                        if (sm_state == SM_IDLE) begin
                            sm_state   <= SM_TO_N2;
                            servo1_reg <= BYTE_UP;
                            dur_cnt    <= DUR_N2;
                        end
                    end
                    CMD_GO_HOLD: begin
                        if (sm_state == SM_AT_N1) begin
                            sm_state   <= SM_LIFTING;
                            servo1_reg <= BYTE_UP;
                            dur_cnt    <= DUR_N1_HOLD;
                        end else if (sm_state == SM_AT_N2) begin
                            sm_state   <= SM_LIFTING;
                            servo1_reg <= BYTE_UP;
                            dur_cnt    <= DUR_N2_HOLD;
                        end
                    end
                    CMD_GO_DOWN: begin
                        if (sm_state == SM_HOLD) begin
                            sm_state   <= SM_LOWERING;
                            servo1_reg <= BYTE_DOWN;
                            dur_cnt    <= DUR_DOWN;
                        end
                    end
                endcase
            end

        endcase

    // ── Prioridad 2: timer expirado (flag registrado) ────────
    // Fix F1: este bloque se ejecuta cuando timer_fired=1 y
    // NO hubo spi_new en este mismo ciclo. Como timer_fired se
    // setea un ciclo antes de que dur_cnt llegue a 0, la SM
    // siempre transiciona incluso si spi_new coincidió con la
    // última cuenta del timer.
    end else if (timer_fired) begin
        timer_fired <= 1'b0;
        servo1_reg  <= BYTE_STOP;
        dur_cnt     <= 26'd0;
        case (sm_state)
            SM_MAN_UP,
            SM_MAN_DOWN: sm_state <= SM_IDLE;
            SM_TO_N1:    sm_state <= SM_AT_N1;
            SM_TO_N2:    sm_state <= SM_AT_N2;
            SM_LIFTING:  sm_state <= SM_HOLD;
            SM_LOWERING: sm_state <= SM_IDLE;
            default:     sm_state <= SM_IDLE;
        endcase
    end

end

// ============================================================
// 7.  MISO — devuelve sm_state durante la transacción SPI
//
//  MODE 0: MISO cambia en flanco de bajada de SCK,
//          Jetson muestrea en flanco de subida.
//  Cargamos sm_state al inicio del CS (cs_fall).
//  Desplazamos MSB-first en cada sck_fall.
//
//  NOTA: el estado devuelto es el de ANTES de procesar el
//  comando del frame actual (cs_fall ocurre antes que cs_rise).
//  Esto es intencional y documentado en el protocolo.
// ============================================================
reg [7:0] miso_sr;

always @(posedge clk) begin
    if (cs_fall)
        miso_sr <= {3'd0, sm_state};      // carga estado actual
    else if (!cs_r && sck_fall)
        miso_sr <= {miso_sr[6:0], 1'b0};  // desplaza MSB first
end

assign miso = cs_r ? 1'b0 : miso_sr[7];

// ============================================================
// 8.  Salidas PWM
// ============================================================
wire [19:0] pw1 = PULSE_MIN + ({12'd0, servo1_reg} * PULSE_STEP);
wire [19:0] pw2 = PULSE_MIN + ({12'd0, servo2_reg} * PULSE_STEP);

assign pwm1     = (pwm_cnt < pw1);
assign pwm2     = (pwm_cnt < pw2);
assign cs_debug = cs_r;

endmodule