/**
 * Tang Nano 20K - Controlador PWM de 2 servos MG90S 360° vía UART
 * CLK: 27 MHz | UART: 115200 8N1 | PWM: 50Hz
 * Protocolo: "S1:90 S2:45\n"  (valores 0-180 mapeados a velocidad)
 * 0   = velocidad máxima CW
 * 90  = parado
 * 180 = velocidad máxima CCW
 */
module tang_servo_controller (
    input  clk,
    input  rst_n,
    input  uart_rx,
    output uart_tx,
    output servo1_pwm,
    output servo2_pwm
);

    // =========================================================
    // Parámetros de temporización
    // =========================================================
    parameter  CLK_FREQ  = 27_000_000;
    parameter  UART_BAUD = 115_200;
    parameter  PWM_FREQ  = 50;

    localparam PWM_PERIOD   = CLK_FREQ / PWM_FREQ;          // 540 000 ciclos = 20ms

    // MG90S 360°: 600µs–2400µs
    localparam DUTY_MIN     = CLK_FREQ / 1_667;             // ~16 200 ciclos = 600µs
    localparam DUTY_MAX     = CLK_FREQ / 417;               // ~64 700 ciclos = 2400µs
    localparam DUTY_RANGE   = DUTY_MAX - DUTY_MIN;          // ~48 500 ciclos
    localparam DUTY_NEUTRAL = (DUTY_MIN + DUTY_MAX) / 2;    // ~40 450 ciclos = 1500µs

    // =========================================================
    // UART RX
    // =========================================================
    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(
        .CLK_FREQ (CLK_FREQ),
        .BAUD_RATE(UART_BAUD)
    ) u_uart_rx (
        .clk  (clk),
        .rst_n(rst_n),
        .rx   (uart_rx),
        .data (rx_data),
        .valid(rx_valid)
    );

    // =========================================================
    // Parser UART + Watchdog (un solo bloque always)
    // =========================================================
    // servo_cmd[n] guarda el valor 0-180 recibido para cada servo
    reg [7:0]  servo_cmd [0:1];

    reg [2:0]  pstate;          // Estado del parser
    reg [0:0]  cur_idx;         // Índice servo actual (0 ó 1)
    reg [7:0]  cur_angle;       // Acumulador de dígitos

    // Watchdog: 27 000 000 / 2 = 13 500 000 → ~0.5s sin datos centra servos
    localparam WD_LIMIT = CLK_FREQ / 2;
    reg [23:0] wd_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pstate      <= 0;
            cur_idx     <= 0;
            cur_angle   <= 0;
            servo_cmd[0]<= 90;
            servo_cmd[1]<= 90;
            wd_cnt      <= 0;

        end else begin

            // --- Watchdog -------------------------------------------
            if (rx_valid) begin
                wd_cnt <= 0;                    // Reset al recibir cualquier byte
            end else if (wd_cnt < WD_LIMIT) begin
                wd_cnt <= wd_cnt + 1;
            end else begin
                servo_cmd[0] <= 90;             // Centra ambos servos
                servo_cmd[1] <= 90;
                pstate       <= 0;
            end

            // --- Parser ---------------------------------------------
            if (rx_valid) begin
                case (pstate)

                    // Espera 'S'
                    0: if (rx_data == "S") pstate <= 1;

                    // Número de servo: '1' o '2'
                    1: begin
                        if (rx_data >= "1" && rx_data <= "2") begin
                            cur_idx   <= rx_data - "1";
                            pstate    <= 2;
                        end else pstate <= 0;
                    end

                    // Espera ':'
                    2: begin
                        if (rx_data == ":") begin
                            cur_angle <= 0;
                            pstate    <= 3;
                        end else pstate <= 0;
                    end

                    // Acumula dígitos del ángulo
                    3: begin
                        if (rx_data >= "0" && rx_data <= "9") begin
                            cur_angle <= cur_angle * 10 + (rx_data - "0");
                        end else if (rx_data == " ") begin
                            // FIX: guarda el ángulo y espera el siguiente 'S'
                            servo_cmd[cur_idx] <= cur_angle;
                            pstate             <= 1;    // ← antes era 0, no avanzaba
                        end else if (rx_data == "\n") begin
                            servo_cmd[cur_idx] <= cur_angle;
                            pstate             <= 0;    // Fin de trama
                        end else begin
                            pstate <= 0;
                        end
                    end

                    default: pstate <= 0;
                endcase
            end
        end
    end

    // =========================================================
    // Cálculo de duty cycle (FIX: operandos en 32 bits)
    // duty = DUTY_MIN + (angle * DUTY_RANGE) / 180
    // Antes: servo_cmd[n] era 8 bits → overflow silencioso
    // Ahora: cast explícito a 32 bits antes de multiplicar
    // =========================================================
    wire [31:0] duty1 = DUTY_MIN + (32'(servo_cmd[0]) * DUTY_RANGE) / 180;
    wire [31:0] duty2 = DUTY_MIN + (32'(servo_cmd[1]) * DUTY_RANGE) / 180;

    // =========================================================
    // Generadores PWM
    // =========================================================
    pwm_gen #(.PERIOD(PWM_PERIOD)) u_pwm1 (
        .clk    (clk),
        .rst_n  (rst_n),
        .duty   (duty1),
        .pwm_out(servo1_pwm)
    );

    pwm_gen #(.PERIOD(PWM_PERIOD)) u_pwm2 (
        .clk    (clk),
        .rst_n  (rst_n),
        .duty   (duty2),
        .pwm_out(servo2_pwm)
    );

    assign uart_tx = 1'b1;

endmodule

// =========================================================
// Módulo PWM genérico (sin cambios)
// =========================================================
module pwm_gen #(
    parameter PERIOD = 540_000
)(
    input            clk,
    input            rst_n,
    input  [31:0]    duty,
    output reg       pwm_out
);
    reg [31:0] cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt     <= 0;
            pwm_out <= 0;
        end else begin
            cnt     <= (cnt >= PERIOD - 1) ? 0 : cnt + 1;
            pwm_out <= (cnt < duty);
        end
    end
endmodule