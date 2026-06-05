// ============================================================
//  spi_slave_lift.v  —  Tang Nano 20K (GW2AR-18C) @ 27 MHz
//  Dual 360° Servo PWM + State Machine + MISO feedback
//
//  MISO: durante cada transacción SPI, el FPGA devuelve su
//  sm_state actual (1 byte). La Jetson lo lee en xfer2().
//  Sin SPI activo, MISO = 0.
//
//  Comandos 0xAC:
//    0xFF  CMD_PING  → no cambia estado, solo devuelve sm_state
//    0x00  STOP      0x20 MAN_UP   0x21 MAN_DOWN
//    0x10  GO_N1     0x11 GO_N2
//    0x12  GO_HOLD   0x13 GO_DOWN
//
//  sm_state en MISO (byte 0 devuelto):
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
wire mosi_r   = mosi_s[1];
wire sck_rise = (sck_s[2:1] == 2'b01);
wire sck_fall = (sck_s[2:1] == 2'b10);   // nuevo: para MISO
wire cs_fall  = (cs_s[2:1]  == 2'b10);
wire cs_rise  = (cs_s[2:1]  == 2'b01);

// ============================================================
// 3.  Receptor SPI (shift register deslizante)
// ============================================================
reg [23:0] shift_reg;
reg [ 2:0] bit_cnt;

always @(posedge clk) begin
    if (cs_fall)        bit_cnt <= 3'd0;
    else if (!cs_r && sck_rise) bit_cnt <= bit_cnt + 3'd1;
    if (!cs_r && sck_rise)
        shift_reg <= {shift_reg[22:0], mosi_r};
end

wire spi_new = cs_rise && (bit_cnt == 3'd0);

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

wire timer_done = sm_moving && (dur_cnt == 26'd1);

// ============================================================
// 6.  SM + decodificador SPI
// ============================================================
always @(posedge clk) begin

    if (sm_moving && dur_cnt > 26'd0)
        dur_cnt <= dur_cnt - 26'd1;

    if (spi_new) begin
        case (shift_reg[23:16])

            8'hAB: begin
                servo1_reg <= shift_reg[15:8];
                servo2_reg <= shift_reg[7:0];
                sm_state   <= SM_IDLE;
                dur_cnt    <= 26'd0;
            end

            8'hAC: begin
                case (shift_reg[15:8])
                    CMD_PING: ; // no-op: solo devuelve sm_state en MISO

                    CMD_STOP: begin
                        sm_state   <= SM_IDLE;
                        servo1_reg <= BYTE_STOP;
                        dur_cnt    <= 26'd0;
                    end
                    CMD_MAN_UP: begin
                        sm_state   <= SM_MAN_UP;
                        servo1_reg <= BYTE_UP;
                        dur_cnt    <= DUR_MAN;
                    end
                    CMD_MAN_DOWN: begin
                        sm_state   <= SM_MAN_DOWN;
                        servo1_reg <= BYTE_DOWN;
                        dur_cnt    <= DUR_MAN;
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
    end

    else if (timer_done) begin
        servo1_reg <= BYTE_STOP;
        dur_cnt    <= 26'd0;
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