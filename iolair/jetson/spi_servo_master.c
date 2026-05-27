/*  spi_servo_master.c
 *  Jetson Nano  →  Tang Nano 20K (SPI slave)  →  2× MG90S 360°
 *
 *  Protocolo (3 bytes por transferencia):
 *    Byte 0 : 0xAB  ← header / identificador de comando
 *    Byte 1 : servo1 (0-255 | 0=reversa max, 127=stop, 255=adelante max)
 *    Byte 2 : servo2 (misma escala)
 *
 *  Pinout SPI en el GPIO header de la Jetson Nano:
 *    Pin 19 – MOSI   (spidev0.0)
 *    Pin 21 – MISO
 *    Pin 23 – SCLK
 *    Pin 24 – CS0    ← /dev/spidev0.0  (Tang Nano CS_N)
 *    Pin 26 – CS1    ← libre por ahora
 *
 *  Compilar:  gcc -O2 -o spi_servo_master spi_servo_master.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

/* ── Configuración SPI ───────────────────────────────────────── */
#define SPI_DEVICE      "/dev/spidev0.0"
#define SPI_MODE        SPI_MODE_0     /* CPOL=0 CPHA=0 — igual que el Verilog */
#define SPI_BITS        8
#define SPI_SPEED_HZ    500000         /* 500 kHz — conservador para FPGA slave */

/* ── Protocolo servo ─────────────────────────────────────────── */
#define CMD_SERVO       0xAB
#define SERVO_STOP      127
#define SERVO_FWD_MAX   255
#define SERVO_REV_MAX   0

static int spi_fd;

/* ── Init / close / transfer (sin cambios vs tu versión original) ── */

int spi_init(void) {
    spi_fd = open(SPI_DEVICE, O_RDWR);
    if (spi_fd < 0) {
        perror("Error abriendo SPI device");
        return -1;
    }

    uint8_t mode = SPI_MODE;
    if (ioctl(spi_fd, SPI_IOC_WR_MODE, &mode) < 0) {
        perror("Error configurando SPI mode");
        close(spi_fd);
        return -1;
    }

    uint8_t bits = SPI_BITS;
    if (ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0) {
        perror("Error configurando bits per word");
        close(spi_fd);
        return -1;
    }

    uint32_t speed = SPI_SPEED_HZ;
    if (ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
        perror("Error configurando speed");
        close(spi_fd);
        return -1;
    }

    return 0;
}

int spi_transfer(uint8_t *tx, uint8_t *rx, uint32_t len) {
    struct spi_ioc_transfer t = {
        .tx_buf        = (unsigned long)tx,
        .rx_buf        = (unsigned long)rx,
        .len           = len,
        .speed_hz      = SPI_SPEED_HZ,
        .bits_per_word = SPI_BITS,
        .delay_usecs   = 0,
    };
    if (ioctl(spi_fd, SPI_IOC_MESSAGE(1), &t) < 0) {
        perror("Error en SPI transfer");
        return -1;
    }
    return 0;
}

void spi_close(void) {
    close(spi_fd);
}

/* ── Helpers de protocolo ─────────────────────────────────────── */

/*  Convierte velocidad (-100 .. +100) al byte que espera el FPGA:
 *   -100  →  0   (reversa máxima, PWM ~1 ms)
 *      0  →  127 (stop,           PWM ~1.5 ms)
 *   +100  →  255 (adelante máx,   PWM ~2 ms)
 */
static uint8_t speed_to_byte(int speed) {
    if (speed < -100) speed = -100;
    if (speed >  100) speed =  100;
    /* mapeo lineal: (-100..100) → (0..255) */
    return (uint8_t)((speed + 100) * 255 / 200);
}

/*  Envía un paquete de 3 bytes para actualizar ambos servos.
 *  s1, s2 en el rango -100 .. +100.
 *  Retorna 0 si OK, -1 si error.
 */
int set_servos(int s1, int s2) {
    uint8_t tx[3] = { CMD_SERVO, speed_to_byte(s1), speed_to_byte(s2) };
    uint8_t rx[3] = { 0 };
    return spi_transfer(tx, rx, 3);
}

/*  Detiene ambos servos — útil como cleanup de emergencia. */
int stop_all(void) {
    return set_servos(0, 0);
}

/* ── main: loop interactivo de prueba ─────────────────────────── */

int main(void) {
    if (spi_init() < 0) {
        fprintf(stderr, "SPI init falló\n");
        return EXIT_FAILURE;
    }

    /* Detener servos al arrancar por seguridad */
    stop_all();

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║   Jetson → Tang Nano 20K  │  Servo SPI Test  ║\n");
    printf("║   Protocolo: [0xAB][s1][s2]  @500 kHz        ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");
    printf("  Velocidad: -100 (reversa max) .. 0 (stop) .. +100 (adelante max)\n");
    printf("  Ingresa 'q' para salir.\n\n");

    char input[32];
    while (1) {
        int s1, s2;

        printf("Servo 1 [-100..100]: ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        if (input[0] == 'q' || input[0] == 'Q') break;
        s1 = atoi(input);

        printf("Servo 2 [-100..100]: ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        if (input[0] == 'q' || input[0] == 'Q') break;
        s2 = atoi(input);

        uint8_t b1 = speed_to_byte(s1);
        uint8_t b2 = speed_to_byte(s2);

        printf("\n─────────────────────────────────────────────\n");
        printf("  TX →  [0x%02X] [0x%02X (%4d)] [0x%02X (%4d)]\n",
               CMD_SERVO, b1, s1, b2, s2);

        if (set_servos(s1, s2) < 0) {
            fprintf(stderr, "  ERROR: SPI transfer falló\n");
            break;
        }

        printf("  S1: %s  │  S2: %s\n",
               s1 == 0 ? "STOP" : (s1 > 0 ? "↑ ADELANTE" : "↓ REVERSA"),
               s2 == 0 ? "STOP" : (s2 > 0 ? "↑ ADELANTE" : "↓ REVERSA"));
        printf("─────────────────────────────────────────────\n\n");
    }

    printf("\nDeteniendo servos y cerrando SPI...\n");
    stop_all();
    spi_close();
    return EXIT_SUCCESS;
}