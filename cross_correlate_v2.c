#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <float.h>
#include <stdbool.h>
#include <inttypes.h>
#include <immintrin.h>
#include <dirent.h>
#include <complex.h>
#include <omp.h>

typedef struct
{
    uint8_t     chunk_id[4];
    uint32_t    chunk_size;
    uint8_t     format[4];
    uint8_t     subchunk1_id[4];
    uint32_t    subchunk1_size;
    uint16_t    audio_format;
    uint16_t    num_channels;
    uint32_t    sample_rate;
    uint32_t    byte_rate;
    uint16_t    block_align;
    uint16_t    bits_per_sample;
    uint8_t     subchunk2_id[4];
    uint32_t    subchunk2_size;

} wav_header;

typedef struct
{
    size_t name_len;
    char* name;

    float freq_guess;
    
    size_t skip_len;
    size_t skip_prds;

    size_t prds;

    size_t data_len;
    float* data;

    float* max;
    size_t* max_idx;

    size_t devs; 

}kern;

uint32_t get_num_samples(wav_header* header)
{
    return 8*header->subchunk2_size/header->num_channels/header->bits_per_sample;
}

void writeKern(char* fn, size_t num_kerns, size_t* kern_lens, float* kern_freq_guess, char** kern_names, float** kerns)
{
    FILE* f = fopen(fn, "wb");

    fwrite(&num_kerns, sizeof(size_t), 1, f);
    fwrite(kern_lens, sizeof(size_t), num_kerns, f);
    fwrite(kern_freq_guess, sizeof(float), num_kerns, f);

    for (int kern_idx = 0; kern_idx < num_kerns; ++kern_idx)
    {   

    	int16_t* data_uint16 = (int16_t*)malloc(sizeof(int16_t)*kern_lens[kern_idx]);
	    for (int smpl_idx = 0; smpl_idx < kern_lens[kern_idx]; ++smpl_idx)
	    {   
	        data_uint16[smpl_idx] = (int16_t)kerns[kern_idx][smpl_idx];
	    }
	    fwrite(data_uint16, sizeof(int16_t), kern_lens[kern_idx], f);

	    free(data_uint16);
	    data_uint16 = NULL;

    }

    fclose(f);
}


size_t gcd(size_t a, size_t b)
{
  size_t c;
  while ( a != 0 ) { c = a; a = b%a;  b = c; }
  return b;
}

size_t findGCD(size_t* arr, size_t n)
{
    size_t result = arr[0];
    for (size_t idx = 1; idx < n; idx++) { result = gcd(arr[idx], result); }

    return result;
}

void readKern(char* fn, size_t* ret_num_kerns, kern* ret_kerns)
{
    FILE* f = fopen(fn, "rb");

    size_t num_kerns;
	fread(&num_kerns, sizeof(size_t), 1, f);

    size_t* kern_lens = malloc(sizeof(size_t)*num_kerns);
    float* kern_freq_guess = malloc(sizeof(float)*num_kerns);

    float** kerns = malloc(sizeof(float*)*num_kerns);

    for (int kern_idx = 0; kern_idx < num_kerns; ++kern_idx) { fread(&kern_lens[kern_idx], sizeof(size_t), 1, f); }
    for (int kern_idx = 0; kern_idx < num_kerns; ++kern_idx) { fread(&kern_freq_guess[kern_idx], sizeof(float), 1, f); }

    for (int kern_idx = 0; kern_idx < num_kerns; ++kern_idx)
    {   

    	int16_t* data_uint16 = (int16_t*)malloc(sizeof(int16_t)*kern_lens[kern_idx]);

        kerns[kern_idx] = (float*)malloc(sizeof(float)*kern_lens[kern_idx]);

        for (int smpl_idx = 0; smpl_idx < kern_lens[kern_idx]; ++smpl_idx) 
        { 
            fread(&data_uint16[smpl_idx], sizeof(int16_t), 1, f); 
            kerns[kern_idx][smpl_idx] = (float)data_uint16[smpl_idx]; 

        }

	    free(data_uint16);
	    data_uint16 = NULL;
    }

    fclose(f);

    ret_kerns = malloc(sizeof(kern)*num_kerns);

    for (int kern_idx = 0; kern_idx < num_kerns; ++kern_idx) 
    {
        ret_kerns[kern_idx].data = (float*)malloc(sizeof(float)*kern_lens[kern_idx]);

        ret_kerns[kern_idx].data = kerns[kern_idx];
        ret_kerns[kern_idx].data_len = kern_lens[kern_idx];
        ret_kerns[kern_idx].freq_guess = kern_freq_guess[kern_idx];
    }

    *ret_num_kerns = num_kerns; 
} 

void readWav(char* fn, wav_header** ret_header, float** data_ret, size_t* ret_num_samples)
{
    FILE* f = fopen(fn, "rb");
    if (!f)
    {
        fprintf(stderr, "Warning! Could not open file: %s\n", fn);
        *ret_header = NULL;
        *data_ret = NULL;
        ret_num_samples = NULL;
        return;
    }

    wav_header* header = malloc(sizeof(wav_header));
    fread(header, sizeof(wav_header), 1, f);

    size_t num_samples = get_num_samples(header);

    if (header->bits_per_sample != 16)
    {
        fprintf(stderr, "Warning! Only supports 16 bits per sample\n");
        return;
    }

    int16_t* data = (int16_t*)malloc(sizeof(int16_t)*header->subchunk2_size);

    fread(data, 1, header->subchunk2_size, f);

    int num_channels = (int)header->num_channels;
    float* out_data = (float*)malloc(sizeof(float)*num_samples*num_channels);

    float max_val = 0.0f;

    for (size_t i = 0; i < num_samples; ++i)
    {
        for (size_t j = 0; j < num_channels; ++j)
        {
            out_data[i*num_channels+j] = (float)data[i*num_channels+j];
            max_val = (fabs(out_data[i*num_channels+j]) > max_val) ? fabs(out_data[i*num_channels+j]) : max_val;
        }
    }

    *data_ret = out_data;
    *ret_num_samples = num_samples;
    *ret_header = header;

    fclose(f);
}

void printFloats(char* output_filename, int srt_line, int end_line, float* data, size_t num_cols)
{
    int num_lines = (end_line - srt_line);

    if (output_filename != NULL)
    {

        FILE* f = fopen(output_filename, "w");
        for (size_t line_idx = srt_line; line_idx < (srt_line + num_lines); ++line_idx)
        {
            for (size_t col_idx = 0; col_idx < num_cols; ++col_idx)
            {
                fprintf(f, "%f,", data[line_idx*num_cols + col_idx]);
            }

            fprintf(f, "%s\n");
        }

        fclose(f);
    }
    else printf("No output selected. Canceling printing. \n" );
}

void printSpect(char* output_filename, float** spect, size_t* kern_devs, size_t num_kerns)
{
    if (output_filename != NULL)
    {
        FILE* f = fopen(output_filename, "w");
        for (size_t kern_idx = 0; kern_idx < num_kerns; ++kern_idx)
        {
            for (size_t dev_idx = 0; dev_idx < kern_devs[kern_idx]; ++dev_idx)
            {
                fprintf(f, "%f,", spect[kern_idx][dev_idx]);
            }

            fprintf(f, "%s\n");
        }

        fclose(f);
    }
    else printf("No output selected. Canceling printing. \n" );
}

int main()
{
	size_t num_kerns;
	kern* kerns; 
	char* kern_name = "./kernal/Piano.kern";
    char* track_name = "./tracks/c_major_scale.wav";

    readKern(kern_name, &num_kerns, kerns);

    wav_header* header;

    float* tracks;
    size_t track_len;

    readWav(track_name, &header, &tracks, &track_len);

    float** spect = malloc(sizeof(float*)*num_kerns);

    for (size_t kern_idx = 0; kern_idx < num_kerns; kern_idx++)
    {   

        kerns[kern_idx].devs = floor(track_len/kerns[kern_idx].data_len);
        spect[kern_idx] = calloc(kerns[kern_idx].devs, sizeof(float)*kerns[kern_idx].devs);

        printf("Freq %f \n", kerns[kern_idx].freq_guess);
        printf("Devs %zu\n", kerns[kern_idx].devs);

        #pragma omp parallel for
        for (size_t dev_idx = 0; dev_idx < kerns[kern_idx].devs; dev_idx++)
        {
            for (size_t smpl_idx = 0; smpl_idx < kerns[kern_idx].data_len; smpl_idx++)
            {
                spect[kern_idx][dev_idx] += kerns[kern_idx].data[smpl_idx]*tracks[dev_idx*kerns[kern_idx].data_len + smpl_idx];

            }

        }
        printf("Kern Percent: %f % \n",(float)kern_idx/(float)num_kerns);

        char filename[512];
        sprintf(filename, "./spect/spectrogram_%f.csv", kerns[kern_idx].freq_guess);

        printFloats(filename, 0, kerns[kern_idx].devs, spect[kern_idx], 1);

    }

    float* lens = malloc(sizeof(float)*num_kerns);

    for (size_t kern_idx = 0; kern_idx < num_kerns; kern_idx++)
    {    
        lens[kern_idx] = (float)kerns[kern_idx].data_len;
    }

    //printSpect("Piano.csv", spect, kern_devs, num_kerns);

    printFloats("kern_lens.csv", 0, num_kerns, lens, 1);

    printf("Track_len: %zu \n", track_len);

    FILE * gnuplotPipe = popen ("gnuplot -persist", "w");
    fprintf(gnuplotPipe, "set terminal png size 1000,1000 \n");
    fprintf(gnuplotPipe, "set output './hello.png' \n");   
    fprintf(gnuplotPipe, "plot './spect/spectrogram_293.978699.csv' with lines \n");      
   
    /*

    fprintf(gnuplotPipe, "plot '-' \n");
    int i;

    for (int i = 0; i < NUM_POINTS; i++)
    {
        fprintf(gnuplotPipe, "%g %g\n", xvals[i], yvals[i]);
    }

    fprintf(gnuplotPipe, "e\n");

    */

    fflush(gnuplotPipe);
    fclose(gnuplotPipe);
    printf("Program completed");

	return 0;
}

/* To do:

- Implement structures
- Work out way to atually plot spectrograms
-



*/