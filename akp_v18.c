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

//^ Most of these are currently unused by the program, just remnants. I'll sort them out at some point

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Structures  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

typedef struct
{
	size_t num_chnls;
	float smpl_rate;
	uint16_t bit_dpth;

	float mag_filter;
	float consec_o;
	float start_skip;
	float max_freq;
	size_t freq_smpl_len_o;
	
	size_t kern_prds;
	size_t skip_prds;

	char* src_ext;
	char* ins_ext;
	char* grp_ext;
	char* src_folder;
	char* kern_folder;

}config_s;

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

}wav_header;

typedef struct
{
	size_t name_len;
	char* name;

	wav_header* header; 

	float freq_guess;

	size_t data_len;
	float* data;

	float* max;
	size_t* max_idx;

}src;

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

}kern;

typedef struct
{
	float freq;
	int idx;

} freq_elm;

typedef struct
{
	float mag;
	int idx;

} mag_elm;


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Input Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

void get_argf(float* val, char* delimeter)
{
	char* t = strtok(NULL, delimeter);
	*val = atof(t);
}

void get_args(char** val, char* delimeter)
{
	char* t = strtok(NULL, delimeter);
	*val = t;
}

void get_argi(int* val, char* delimeter)
{
	char* t = strtok(NULL, delimeter);
	*val = atoi(t);
}

void get_argzu(size_t* val, char* delimeter)
{
	char* t = strtok(NULL, delimeter);
	*val = atoi(t);
}

void readFileFloat(char* file_name, char* delimeter, size_t num_cols, size_t srt_line, size_t srt_col, float** data_ret, size_t* lines_read)
{

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Reads data from input file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	size_t init_num_lines = 100;
	size_t curr_num_lines = init_num_lines;

	float* data = malloc(sizeof(float)*num_cols*init_num_lines);

	char* buffer = NULL;

	size_t size;

	FILE* file = fopen(file_name, "r");

	if (file == NULL)
	{
		printf("Could not find input file! \"%s\"\n", file_name);
		return;
	}

	//Skips to start line:
	for (size_t line_idx = 0; line_idx < srt_line; ++line_idx) { getline(&buffer, &size, file); }

	size_t line_idx = 0;

	while (getline(&buffer, &size, file) != EOF)
	{
		if(line_idx >= curr_num_lines)
		{
			data = realloc(data, sizeof(float)*(curr_num_lines + init_num_lines)*num_cols);
			curr_num_lines += init_num_lines;
		}

		char* first = strtok(buffer, delimeter);
		data[line_idx*num_cols + 0] = atof(first);

		//Skips to start column
		float empty = 0;
		for (size_t col_idx = 1; col_idx < srt_col; ++col_idx) get_argf(&empty, delimeter);

		if (srt_col == 0) { for (size_t col_idx = 1; col_idx < num_cols; ++col_idx) { get_argf(&data[line_idx*num_cols + col_idx], delimeter); } }
		else              { for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) { get_argf(&data[line_idx*num_cols + col_idx], delimeter); } } 

		line_idx++;
	}

	*lines_read = line_idx;
	*data_ret = data;

	fclose(file);
}

// ~~~~~~~ Reading Wavs ~~~~~~~ //

uint32_t get_num_samples(wav_header* header)
{
    return 8*header->subchunk2_size/header->num_channels/header->bits_per_sample;
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

// ~~~~~~~ File name stuff ~~~~~~~ //

void readDirContents(char* dir_name, char*** strings_ret, size_t* num_strings)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Gets the names of all files within dir_name, and stores them in stings_ret. Number of files found is stored in num_strings.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	int init_num_strings = 100;
	int curr_num_strings = init_num_strings;
	char** strings = malloc(sizeof(char*)*init_num_strings);

	DIR* dir = opendir(dir_name);
	if (dir == NULL)
	{
		fprintf(stderr, "Could not open directory! %s\n", dir_name);
		exit(1);
	}

	int name_idx = 0;

	struct dirent* ent;
	while((ent = readdir(dir)) != NULL)
	{
		if(name_idx >= curr_num_strings)
		{
			strings = realloc(strings, sizeof(char*)*curr_num_strings*2);
			curr_num_strings *= 2;
		}

		int name_sz = strlen(ent->d_name)+1;
		strings[name_idx] = malloc(sizeof(char)*name_sz);

		strcpy(strings[name_idx], ent->d_name);

		name_idx++;
	}

	if (name_idx == 0)
	{
		*strings_ret = strings;
		*num_strings = 0;
		return;
	}

	*strings_ret = strings;
	*num_strings = name_idx;
	
}

void filterbyExtension(const char* ext, char** input, size_t num_input, char*** filt_ret, size_t* num_filt)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Filters a list of filenames, imput, separating the files with extension, ext, into a new list filt_ret. Number of items retained after filtration being given by num_filt. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	if (num_input == 0)
	{
		*filt_ret = NULL;
		*num_filt = 0;
		return;
	}

	size_t filt_idx = 0;
	const char* dot_pos;
	int check_ext;

	int init_num_filt = 100;
	int curr_num_filt = init_num_filt;
	char** filt = malloc(sizeof(char*)*init_num_filt);

	for (size_t input_idx = 0; input_idx < num_input; ++input_idx)
	{
		dot_pos = strrchr(input[input_idx], '.');
		if (!dot_pos || dot_pos == input[input_idx]) { continue; }
		else 
		{
			check_ext = strcmp(dot_pos+1, ext);
			if (check_ext == 0) 
			{	
				if(filt_idx >= curr_num_filt)
				{
					filt = realloc(filt, sizeof(char*)*curr_num_filt*2);
					curr_num_filt *= 2;
				}

				int filt_sz = strlen(input[input_idx])+1;
				filt[filt_idx] = malloc(sizeof(char)*filt_sz);

				strcpy(filt[filt_idx],input[input_idx]);

				filt_idx++;
			}

		}
	}

	if (num_filt == 0)
	{
		*filt_ret = NULL;
		*num_filt = 0;

		return;
	}

	*filt_ret = filt;
	*num_filt = filt_idx;
}

void readSrcs(char* dir_name, config_s config, src** srcs_ret, size_t* num_srcs_ret)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Reads all src files with extension ext within dir_name. src files must have num_chnls indicated.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	char** file_names;
	size_t num_files;

	size_t num_srcs_o;
	size_t num_srcs;

	char dir_name_temp[(strlen(dir_name) + strlen(config.src_folder) + strlen(config.ins_ext) + 3)];
	sprintf(dir_name_temp, "%s%s.%s", config.src_folder, dir_name, config.ins_ext);

	readDirContents(dir_name_temp, &file_names, &num_files);

	if (num_files == 0)
	{
		printf("Warning: No source files detected! \n");
		free(file_names);
		*num_srcs_ret = 0;
		*srcs_ret = NULL;
		return;
	}

	char** src_names;
	filterbyExtension(config.src_ext, file_names, num_files, &src_names, &num_srcs_o);

	src* srcs_o = malloc(sizeof(src)*num_srcs_o);
	src* srcs = malloc(sizeof(src)*num_srcs_o);;

	for (size_t src_idx = 0; src_idx < num_srcs_o; ++src_idx)
	{
		srcs_o[src_idx].name = malloc(sizeof(src_names[src_idx]));
		srcs_o[src_idx].name_len = strlen(src_names[src_idx]);
		srcs_o[src_idx].name = src_names[src_idx];
		srcs_o[src_idx].name[srcs_o[src_idx].name_len - strlen(config.src_ext) - 1] = '\0';
	}

	num_srcs = num_srcs_o;

	if (num_srcs_o == 0)
	{
		printf("Warning: No source files detected! \n");
		*num_srcs_ret = 0;
		*srcs_ret = NULL;
		return;
	}

	size_t* check_idx = malloc(sizeof(size_t)*num_srcs_o);
	size_t idx = 0;

	for (size_t src_idx = 0; src_idx < num_srcs_o; ++src_idx)
	{
		char src_name_temp[(strlen(srcs_o[src_idx].name) + strlen(dir_name_temp) + strlen(config.src_ext) + 4)];
		sprintf(src_name_temp, "%s/%s.%s", dir_name_temp, srcs_o[src_idx].name, config.src_ext);

		srcs_o[src_idx].header = malloc(sizeof(wav_header));

		readWav(src_name_temp, &srcs_o[src_idx].header, &srcs_o[src_idx].data, &srcs_o[src_idx].data_len);

		if((&srcs_o[src_idx].data_len == NULL || srcs_o[src_idx].header->num_channels != config.num_chnls) || (srcs_o[src_idx].header->bits_per_sample != config.bit_dpth))
		{
			printf("Warning: Source invalid! Skipping! %s\n", srcs_o[src_idx].name);
			num_srcs -= 1;
			continue;
		}
		else
		{
			check_idx[idx] = src_idx; 
			idx += 1;
		}

	}

	for (size_t src_idx = 0; src_idx < idx; ++src_idx)
	{
		srcs[src_idx] = srcs_o[check_idx[src_idx]];	
	}

	*num_srcs_ret = num_srcs;
	*srcs_ret = srcs;

	free(srcs_o);
	free(check_idx);
	free(file_names);
	free(src_names);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Kernal Processing Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

// ~~~~~~~ Fourrier Stuff ~~~~~~~ //


void _calcFFT(float complex* input, float complex* buf, int n, int step)
{
	if (step >= n) return;
	_calcFFT(buf, input, n, step*2);
	_calcFFT(buf+step, input+step, n, step*2);

	for (int k = 0; k < n; k += 2*step)
	{
		float complex w = cexpf(-I*M_PI*k/n);
		input[k/2] 	= buf[k] + w*buf[k+step];
		input[(k+n)/2] = buf[k] - w*buf[k+step];
	}
}

void calcFFT(float complex* input, float complex* output, int n)
{
	float complex* buf = malloc(sizeof(float complex)*n);
	for (int i = 0; i < n; ++i) 
	{
		buf[i] = input[i];
		output[i] = input[i];
	}
	_calcFFT(output, buf, n, 1);
}

bool checkPower2(size_t x) { return x && !(x & (x - 1));}

int nearestPower2(int x) {return pow(2, floor(log(x)/log(2)));}

int mag_elm_cmp(const void* a, const void* b)
{
	mag_elm* ax = (mag_elm*)a;
	mag_elm* bx = (mag_elm*)b;
	if (ax->mag < bx->mag) return 1;
	if (ax->mag == bx->mag) return 0;
	if (ax->mag > bx->mag) return -1; 
}

int mag_elm_idx(const void* a, const void* b)
{
	mag_elm* ax = (mag_elm*)a;
	mag_elm* bx = (mag_elm*)b;
	if (ax->idx < bx->idx) return 1;
	if (ax->idx == bx->idx) return 0;
	if (ax->idx > bx->idx) return -1; 
}

void calcFFTMag(int n, float complex* input, mag_elm** mag_arr_ret)
{
	if (checkPower2(n) == 0)
	{
		printf("Warning; not a power of 2!");
		return;
	}

	float complex* output = malloc(sizeof(float complex)*n);
	
	calcFFT(input, output, n);

	mag_elm* mag_arr = malloc(sizeof(mag_elm)*n);

	//Creating magnitude structure:

	for (int i = 0; i < n/2+1; ++i)
	{
		mag_arr[i].mag = sqrtf(creal(output[i])*creal(output[i]) + cimag(output[i])*cimag(output[i]));
		mag_arr[i].idx = i;
	}

	*mag_arr_ret = mag_arr;
}

// ~~~~~~~ Other kernel generation functions ~~~~~~~ //


void guessFundFreq(size_t n, config_s config, float complex* input, float* fund_freq_guess)
{

	float mag_filter = config.mag_filter; float consec_o = config.consec_o; float start_skip = config.start_skip; 

	mag_filter *= n;
	consec_o *= n;
	start_skip *= n;

	size_t mag_filter_int = (size_t)mag_filter;
	size_t consec_o_int = (size_t)consec_o;
	size_t start_skip_int = (size_t)start_skip;
	
	size_t consec = consec_o_int;

	float min_freq = config.max_freq;
	size_t min_freq_idx = 0;
	float ampl_min_freq = 0.0f;

	//Calculate FFT magnitude array:
	mag_elm* mag_arr = malloc(sizeof(mag_elm)*n);
	calcFFTMag(n, input, &mag_arr);

	//Sort by magnitude:

	qsort(mag_arr, n/2+1, sizeof(mag_arr), mag_elm_cmp);

	//Create new array of top (mag_filter) num of elements:

	mag_elm* top_mag_arr = malloc(sizeof(mag_elm)*mag_filter_int);
	for (size_t i = 0; i < mag_filter_int; ++i) { top_mag_arr[i] = mag_arr[i]; }

	free(mag_arr);

	//Sort top mag values by idx:

	//qsort(top_mag_arr, mag_filter_int, sizeof(top_mag_arr), mag_elm_idx);

	//Apply square window and find lowest peak:

	for (size_t i = 0; i < mag_filter_int; ++i)
	{
		float freq = top_mag_arr[i].idx * config.smpl_rate/n;
		float ampl = top_mag_arr[i].mag * 2.0f/n;
		if (freq > start_skip && freq < min_freq)
		{
			if (min_freq_idx - top_mag_arr[i].idx <= consec && ampl < ampl_min_freq) 
			{ 
				//If lowest values are consecutive, and not higher than before, skip.
				consec = min_freq_idx - top_mag_arr[i].idx + consec_o_int;
				continue; 
			}
			else 
			{
			min_freq = freq;
			ampl_min_freq = ampl;
			min_freq_idx = top_mag_arr[i].idx;
			consec = consec_o_int;
			}
		}
	}

	free(top_mag_arr);

	*fund_freq_guess = min_freq;
}

void calcMax(float* array, size_t num_lines, size_t num_cols, float* max, size_t* max_idx)
{
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Calculates the maximum value of each column within an array.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //


	for (size_t col_idx = 0; col_idx < num_cols; ++col_idx)
	{
		float temp_max = array[0]; 
		size_t temp_max_idx = 0;

		for (size_t line_idx = 0; line_idx < num_lines; ++line_idx)
		{
			if (array[line_idx*num_cols + col_idx] > temp_max) 
			{ 
				temp_max = array[line_idx*num_cols + col_idx]; 
				temp_max_idx = line_idx;
			}
		}

		max[col_idx] = temp_max;
		max_idx[col_idx] = temp_max_idx;

	}
}

// ~~~~~~~ Windowing functions ~~~~~~~ //

void generateHamm(size_t intervals, float** data_ret)
{
	float* data = malloc(sizeof(float)*intervals);
	for (size_t idx = 0; idx < intervals; ++idx)
	{
		data[idx] = 0.54 - 0.46*cos((2*M_PI*idx)/(intervals-1));
	}

	*data_ret = data;
}

void generateHann(size_t intervals, float** data_ret)
{
	float* data = malloc(sizeof(float)*intervals);
	for (size_t idx = 0; idx < intervals; ++idx)
	{
		data[idx] = sin((M_PI*idx)/(intervals-1))*sin((M_PI*idx)/(intervals-1));
	}

	*data_ret = data;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Output Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

int compKernFreq(const void* a, const void* b)
{
	kern* ax = (kern*)a;
	kern* bx = (kern*)b;
	if (ax->freq_guess < bx->freq_guess) return 1;
	if (ax->freq_guess == bx->freq_guess) return 0;
	if (ax->freq_guess > bx->freq_guess) return -1; 
}


void printFloats(char* output_filename, int srt_line, int end_line, float* data, size_t num_cols, int print_number)
{
	int num_lines = (end_line - srt_line);

	if (output_filename != NULL)
	{

		char filename_positions[512];
   		sprintf(filename_positions, "%s_test_%d.csv", output_filename, print_number);

		FILE* f = fopen(filename_positions, "w");
		for (size_t line_idx = srt_line; line_idx < (srt_line + num_lines); ++line_idx)
		{
			for (size_t col_idx = 0; col_idx < num_cols; ++col_idx)
			{
				fprintf(f, "%f", data[line_idx*num_cols + col_idx]);
			}

			fprintf(f, "%s\n");

		}

		fclose(f);
	}
	else printf("No output selected. Canceling printing. \n" );
}

void writeKern(char* fn, size_t num_kerns, kern* kerns)
{
    FILE* f = fopen(fn, "wb");

    fwrite(&num_kerns, sizeof(size_t), 1, f);

    for (int kern_idx = 0; kern_idx < num_kerns; ++kern_idx)
    {   
	    fwrite(&kerns[kern_idx], sizeof(kern), 1, f);
    }

    fclose(f);
}

void writeKernGrp(char* grp_name, config_s config)
{

	size_t num_srcs;
	src* srcs; 

	readSrcs(grp_name, config, &srcs, &num_srcs);
	printf("Number of sources: %i\n", num_srcs);

	size_t num_kerns = num_srcs*config.num_chnls;
	kern* kerns = malloc(sizeof(kern)*num_srcs*config.num_chnls); 

	#pragma omp parallel for
	for (size_t src_idx = 0; src_idx < num_srcs; ++src_idx) 
	{
		size_t freq_smpl_len = config.freq_smpl_len_o;

		srcs[src_idx].max = malloc(sizeof(float)*config.num_chnls); srcs[src_idx].max_idx = malloc(sizeof(size_t)*config.num_chnls);
		calcMax(srcs[src_idx].data, srcs[src_idx].data_len, config.num_chnls, srcs[src_idx].max, srcs[src_idx].max_idx);

		for (size_t chnl_idx = 0; chnl_idx < config.num_chnls; ++chnl_idx) 		
		{ 
			size_t kern_idx = src_idx*config.num_chnls + chnl_idx;

			kerns[kern_idx].skip_prds = config.skip_prds;
			kerns[kern_idx].prds = config.kern_prds; 

			kerns[kern_idx].name = malloc(sizeof(char)*(strlen(srcs[src_idx].name) + 20));
			if (config.num_chnls > 1) {sprintf(kerns[kern_idx].name, "%s_chnl_%zu", srcs[src_idx].name, chnl_idx);}
			else               {sprintf(kerns[kern_idx].name, "%s", srcs[src_idx].name);}

			if (freq_smpl_len + srcs[src_idx].max_idx[chnl_idx] > srcs[src_idx].data_len)
			{
				freq_smpl_len = nearestPower2(srcs[src_idx].data_len - srcs[src_idx].max_idx[chnl_idx]);

			}
			else
			{
				freq_smpl_len = config.freq_smpl_len_o;
			}
	
			complex float* freq_smpl_cplx = malloc(sizeof(complex float) * freq_smpl_len);

			for (size_t smpl_idx = 0; smpl_idx < freq_smpl_len; ++smpl_idx)
			{
				freq_smpl_cplx[smpl_idx] = (complex float)srcs[src_idx].data[srcs[src_idx].max_idx[chnl_idx] + smpl_idx*config.num_chnls];

			}

			guessFundFreq(freq_smpl_len, config, freq_smpl_cplx, &kerns[kern_idx].freq_guess);

			kerns[kern_idx].data_len = floor((config.smpl_rate/kerns[kern_idx].freq_guess)*config.kern_prds);
			kerns[kern_idx].data = malloc(sizeof(float)*kerns[kern_idx].data_len); 

			float* wind;			
			generateHann(kerns[kern_idx].data_len, &wind);

			kerns[kern_idx].skip_len = floor((config.smpl_rate/kerns[kern_idx].freq_guess)*config.skip_prds);

			for (size_t smpl_idx = 0; smpl_idx < kerns[kern_idx].data_len; ++smpl_idx)
			{
				kerns[kern_idx].data[smpl_idx] = srcs[src_idx].data[srcs[src_idx].max_idx[chnl_idx] + kerns[kern_idx].skip_len + smpl_idx]*wind[smpl_idx];
			}

			kerns[kern_idx].max = malloc(sizeof(float));
			kerns[kern_idx].max_idx = malloc(sizeof(size_t)*config.num_chnls);
			calcMax(kerns[kern_idx].data, kerns[kern_idx].data_len, config.num_chnls, kerns[kern_idx].max, kerns[kern_idx].max_idx);

			for (size_t smpl_idx = 0; smpl_idx < kerns[kern_idx].data_len; ++smpl_idx)
			{
				kerns[kern_idx].data[smpl_idx] = kerns[kern_idx].data[smpl_idx]/kerns[kern_idx].max[chnl_idx];
			}

			free(wind);
			free(freq_smpl_cplx);
		}	
	}

	qsort(kerns, num_kerns, sizeof(kern), compKernFreq);

	for (size_t src_idx = 0; src_idx < num_srcs; ++src_idx) 
	{

		for (size_t chnl_idx = 0; chnl_idx < config.num_chnls; ++chnl_idx) 		
		{ 
			size_t kern_idx = src_idx*config.num_chnls + chnl_idx;
			//printFloats(srcs[src_idx].name, 0, kerns[kern_idx].data_len, kerns[kern_idx].data, config.num_chnls, chnl_idx);  // needs chnlginhg for multi_chnls if stil i
		}
	}

	char grp_name_temp[strlen(config.kern_folder) + strlen(grp_name) + strlen(config.grp_ext) + 3];
	sprintf(grp_name_temp, "%s%s.%s", config.kern_folder, grp_name, config.grp_ext);
	printf("Group name: %s\n", grp_name_temp);

	writeKern(grp_name_temp, num_kerns, kerns);

	for (size_t src_idx = 0; src_idx < num_srcs; ++src_idx) 
	{
		free(srcs[src_idx].data);
		free(srcs[src_idx].name);
		free(srcs[src_idx].header);
		free(srcs[src_idx].max);
		free(srcs[src_idx].max_idx);
	}
	free(srcs);

	for (size_t kern_idx = 0; kern_idx < num_kerns; ++kern_idx) 
	{
		free(kerns[kern_idx].data);
		free(kerns[kern_idx].name);
		free(kerns[kern_idx].max);
		free(kerns[kern_idx].max_idx);
	}

	free(kerns);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

int main(int argc, char** argv)
{
	printf("Program Start. \n");

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Config ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	//This does nothing at the moment, but will eventually be used to read in a config file. Will probably have to make a readconfig function rather than an all float one.

	char* config_filename = "config.csv";
	char* config_delimter = ",";

	size_t config_cols = 13; 
	size_t config_srt_line = 0;
	size_t config_srt_col = 0;

	float* config_f;
	size_t lines_read;

	printf("Loading config... \n");

	readFileFloat(config_filename, config_delimter, config_cols, config_srt_line, config_srt_col, &config_f, &lines_read);

	config_s config;

	//Expected source file parameters:
	config.num_chnls = 1;
	config.smpl_rate = 44100.0f;
	config.bit_dpth = 16;

	//Frequency determination parameters:
	config.mag_filter = 0.00150f;
	config.consec_o = 0.000150f;
	config.start_skip = 0.000150f;
	config.max_freq = 10000.0f;
	config.freq_smpl_len_o = nearestPower2(1000000);
	if (checkPower2(config.freq_smpl_len_o) == 0) { printf("Warning not a powerr of 2!"); exit(1); }
	
	//Kernel generation parameters:
	config.kern_prds = 20;
	config.skip_prds = 100;

	//File locations and extensions: 
	config.src_ext = "wav";
	config.ins_ext = "ins";
	config.grp_ext = "kern";
	config.src_folder = "./source/";
	config.kern_folder = "./kernal/";

	printf("Config Loaded. \n");

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Read Audio File ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Kernal Stuff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	//Here I am working on building a function to convert src files to kernal ones, I thought I'd do this on a by kernal group level rather than reading all the srcal groups first. 

	printf("Loading sources. \n");

	char** file_names; size_t num_files; char** grp_names; size_t num_grps;

	readDirContents(config.src_folder, &file_names, &num_files);
	filterbyExtension(config.ins_ext, file_names, num_files, &grp_names, &num_grps);

	printf("Number grps: %zu \n", num_grps);

	for (size_t grp_idx = 0; grp_idx < num_grps; ++grp_idx)
	{
		printf("Kernel group name: %s \n", grp_names[grp_idx]);
	}

	for (size_t grp_idx = 0; grp_idx < num_grps; ++grp_idx)
	{
		size_t len = strlen(grp_names[grp_idx]);
		grp_names[grp_idx][len-4] = '\0';

		printf("Num grps: %s \n", grp_names[grp_idx]);

		writeKernGrp(grp_names[grp_idx], config);
	}

	printf("Program completed");
	return 0;

}