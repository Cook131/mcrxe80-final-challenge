/**
 * Módulo UART RX - 8N1
 * CLK: 27 MHz
 * Baud: configurable (default 115200)
 *
 * Salidas:
 *   data  - byte recibido
 *   valid - pulso de 1 ciclo cuando data es válido
 */
module uart_rx #(
    parameter CLK_FREQ  = 27_000_000,
    parameter BAUD_RATE = 115200
)(
    input            clk,
    input            rst_n,
    input            rx,
    output reg [7:0] data,
    output reg       valid
);

    // Cuántos ciclos dura un bit
    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE; // 234 ciclos @ 27MHz/115200

    // Muestreamos en el centro del bit
    localparam HALF_BIT = CLKS_PER_BIT / 2;

    // Estados del receptor
    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    // Sincronizador de 2 etapas para rx (evita metaestabilidad)
    reg rx_sync0, rx_sync1;
    always @(posedge clk) begin
        rx_sync0 <= rx;
        rx_sync1 <= rx_sync0;
    end

    reg [1:0]  state;
    reg [8:0]  clk_cnt;   // Contador de ciclos dentro del bit
    reg [2:0]  bit_idx;   // Índice del bit actual (0-7)
    reg [7:0]  rx_shift;  // Registro de desplazamiento

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            clk_cnt  <= 0;
            bit_idx  <= 0;
            rx_shift <= 0;
            data     <= 0;
            valid    <= 0;
        end else begin
            valid <= 0; // Pulso de 1 ciclo, se baja por defecto

            case (state)

                // Espera flanco de bajada (inicio de start bit)
                S_IDLE: begin
                    if (!rx_sync1) begin
                        state   <= S_START;
                        clk_cnt <= 0;
                    end
                end

                // Espera hasta el centro del start bit y verifica que sigue en 0
                S_START: begin
                    if (clk_cnt == HALF_BIT) begin
                        if (!rx_sync1) begin
                            // Start bit válido, empezamos a leer datos
                            state   <= S_DATA;
                            clk_cnt <= 0;
                            bit_idx <= 0;
                        end else begin
                            // Fue ruido, regresamos a IDLE
                            state <= S_IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                // Lee 8 bits de datos, muestreando en el centro de cada bit
                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        // Centro del bit: capturamos LSB primero (UART estándar)
                        rx_shift <= {rx_sync1, rx_shift[7:1]};
                        clk_cnt  <= 0;

                        if (bit_idx == 7) begin
                            state <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                // Verifica stop bit y publica el dato
                S_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        if (rx_sync1) begin
                            // Stop bit en alto = frame válido
                            data  <= rx_shift;
                            valid <= 1;
                        end
                        // Si stop bit es 0 = framing error, descartamos silenciosamente
                        state   <= S_IDLE;
                        clk_cnt <= 0;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

            endcase
        end
    end

endmodule