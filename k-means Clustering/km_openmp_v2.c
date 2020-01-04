#define _POSIX_C_SOURCE 199309L

#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>
#include<string.h>
#include<float.h>
#include<math.h>
#include<omp.h>

/* OSX timer includes */
 #ifdef __MACH__
   #include <mach/mach.h>
   #include <mach/mach_time.h>
 #endif

 /**
 * @brief Return the number of seconds since an unspecified tim
 e (e.g., Unix
 *        epoch). This is accomplished with a high-resolution m
 onotonic timer,
 *        suitable for performance timing.
 *
 * @return The number of seconds.
 */
static inline double monotonic_seconds()
{
 #ifdef __MACH__
   /* OSX */
   static mach_timebase_info_data_t info;
   static double seconds_per_unit;
   if(seconds_per_unit == 0) {
     mach_timebase_info(&info);
     seconds_per_unit = (info.numer / info.denom) / 1e9;
   }
   return seconds_per_unit * mach_absolute_time();
 #else
   /* Linux systems */
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + ts.tv_nsec * 1e-9;
 #endif 
}

#define BUFFER_SIZE 1000000
#define ITERATIONS 20

struct cluster_assg_args {
	double **row_pointer;
	int thread_id; 
	int rows;
};

double **cluster_centroids = NULL;
double **global_centroids = NULL;
double ***local_cluster_centroids = NULL;
int **local_cluster_counts = NULL;
int *cluster_assignment_map = NULL;
int converged = 1;
int rows_test = 0;
int dims_test = 0;
int clusters = 0;
int p = 0;



static void print_time(double const seconds) {
	printf("k-means clustering time: %0.04fs\n", seconds);
}

inline double distance(double *v1, double *v2) {

	int j;
	double sq_dist = 0.0;
	for(j = 0; j < dims_test; j++) {
		double cost = v1[j] - v2[j];		
		sq_dist += cost * cost;
	}
	return sq_dist;
}

inline void sum(double *v1, double *v2) {

	int j;
	for(j = 0; j < dims_test; j++) {
		v1[j] += v2[j];
	}
}

int main(int argc, char *argv[]) {

	if(argc <= 3) {
		printf("Please enter command line params in this order: filename number_of_clusters number_of_threads. \n");
		exit(0);
	}

	int i, j, k;
	const char *delim = " \0";
	const char *file = argv[1];
	FILE *input_file = fopen(file, "r");
	char buffer[BUFFER_SIZE]; 
	char *token;
	int rows = 0, dims = 0;
	clusters = strtol(argv[2], NULL, 10);
	p = strtol(argv[3], NULL, 10);
	pthread_t p_threads[p];
	pthread_attr_t attr;

	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

	double **mat = NULL;

	int dims_set = 0;
	int first_line = 1;
	if(input_file == NULL) {
		printf("File open failed!");
		return(1);
	} else {
		while(fgets(buffer, BUFFER_SIZE, input_file) != NULL) {
			token = strtok(buffer, delim);

			if(first_line) {
				
				rows_test = (int) strtol(token, NULL, 10);
		
				while(token != NULL) {
					dims_test = (int) strtol(token, NULL, 10);
					token = strtok(NULL, delim);
				}

				mat = (double **) malloc(rows_test * sizeof(double *));
				int row;
				for(row = 0; row < rows_test; row++) {
					mat[row] = (double *) malloc(dims_test * sizeof(double));
				}

				first_line = 0;
				continue;
			}

			while(token != NULL) {
				mat[rows][dims] = strtod(token, NULL);
				token = strtok(NULL, delim);
				dims++;
			}

			dims_set = 1;
			rows++;
			dims = 0;
		}
				
		if(ferror(input_file)) {
			perror("Error occured: ");
		}

		fclose(input_file);
	}
		
	double start = monotonic_seconds();

	//Final global centroids - allocation and init.
	cluster_centroids = (double **) malloc(clusters * sizeof(double *));
	//Interim global centroids - allocation and init.
	global_centroids = (double **) malloc(clusters * sizeof(double *));
	for(k = 0; k < clusters; k++) {
		cluster_centroids[k] = (double *) malloc(dims_test * sizeof(double));
		global_centroids[k] = (double *) malloc(dims_test * sizeof(double));
	}
	
	for(k = 0; k < clusters; k++) {
		for(j = 0; j < dims_test; j++) { 
			cluster_centroids[k][j] = mat[k][j];
			global_centroids[k][j] = 0.0;
		}	
	}

	//Local threads - allocation. 
	local_cluster_centroids = (double ***) malloc(p * sizeof(double **));
        for(k = 0; k < p; k++) {
		local_cluster_centroids[k] = (double **) malloc(clusters * sizeof(double *));
		for(j = 0; j < clusters; j++) {
			local_cluster_centroids[k][j] = (double *) malloc(dims_test * sizeof(double)); 
		}
	}

	//Local cluster counts - allocate. 
	local_cluster_counts = (int **) malloc(p * sizeof(int *));
        for(k = 0; k < p; k++) {
		local_cluster_counts[k] = (int *) malloc(clusters * sizeof(int));
	}

	//Cluster assignment map.
	cluster_assignment_map = (int *) malloc(rows_test * sizeof(int));

	int iter_count = 0;
	int myid;
	omp_set_num_threads(p);

	while(iter_count < ITERATIONS) {
	
		//Local threads - init.
		for(i = 0; i < p; i++)
			for(j = 0; j < clusters; j++)
				for(k = 0; k < dims_test; k++)
					local_cluster_centroids[i][j][k] = 0.0;

		//Local cluster counts - init.
		for(j = 0; j < p; j++) {
			for(k = 0; k < clusters; k++) {
				local_cluster_counts[j][k] = 0;
			}
		}

		int thread_id;
		int chunk_size = rows_test/p + 1;
		#pragma omp parallel default(none) private(thread_id, k, j) \
			 shared(chunk_size, rows_test, dims_test, mat, clusters, converged, cluster_centroids, local_cluster_centroids, local_cluster_counts, cluster_assignment_map)
		{
			thread_id = omp_get_thread_num();
			int *cluster_counts = (int *) malloc(clusters * sizeof(int));
			double **t_cluster_centroids = (double **) malloc(clusters * sizeof(double *));

			for(k = 0; k < clusters; k++) {
				t_cluster_centroids[k] = (double *) malloc(dims_test * sizeof(double));
				for(j = 0; j < dims_test; j++)
					t_cluster_centroids[k][j] = 0.0;
				cluster_counts[k] = 0;
			}

			#pragma omp for schedule(dynamic, chunk_size)
			for(i = 0; i < rows_test; i++) {

                		double min = DBL_MAX;
                		int cluster_id = 0;
                		for(k = 0; k < clusters; k++) {
                        		double dist = distance(mat[i], cluster_centroids[k]);
                        		if(dist < min) {
                                		min = dist;
                                		cluster_id = k;
                        		}
                		}

				for(j = 0; j < dims_test; j++)
                			t_cluster_centroids[cluster_id][j] += mat[i][j]; 
				               		
				cluster_counts[cluster_id] += 1;
				if(cluster_assignment_map[i] != cluster_id) {
                		        cluster_assignment_map[i] = cluster_id;
                		        converged = 0;
                		}

        		} //end of pragma for

			//assuming the parallel for section has ended by this step.
			for(k = 0; k < clusters; k++) { 
				local_cluster_counts[thread_id][k] += cluster_counts[k];
				for(j = 0; j < dims_test; j++) {
					local_cluster_centroids[thread_id][k][j] += t_cluster_centroids[k][j];
				}
			}

			free(cluster_counts);
			for(k = 0; k < clusters; k++)
				free(t_cluster_centroids[k]);
			free(t_cluster_centroids);
		}//end of parallel section

		for(i = 0; i < p; i++) { //Thread
			for(j = 0; j < clusters; j++) { 
				for(k = 0; k < dims_test; k++) { 
					global_centroids[j][k] += local_cluster_centroids[i][j][k];
				}
				if(i != 0) local_cluster_counts[0][j] += local_cluster_counts[i][j];
			}
		}

		for(i = 0; i < clusters; i++) {
			if(local_cluster_counts[0][i] || !iter_count)
				for(j = 0; j < dims_test; j++) {
					cluster_centroids[i][j] = global_centroids[i][j] / local_cluster_counts[0][i];
				}
			else
				for(j = 0; j < dims_test; j++) {
					cluster_centroids[i][j] = global_centroids[i][j];
				}
		}

		for(i = 0; i < clusters; i++) {
			for(j = 0; j < dims_test; j++) {
				global_centroids[i][j] = 0.0;
			}
		}

		for(i = 0; i < p; i++) {
			for(j = 0; j < clusters; j++) {
				for(k = 0; k < dims_test; k++) {
					local_cluster_centroids[i][j][k] = 0.0;
				}
				local_cluster_counts[i][j] = 0;
			}
		}

		iter_count++;	

		if(converged && iter_count)
			break; 

		converged = 1;

	}//end while - Indent required.

	double time = monotonic_seconds() - start;
	print_time(time);

	FILE *f = fopen("clusters.txt", "w");
	if (f == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}

	for(i = 0; i < rows_test; i++) {
		fprintf(f, "%d\n", cluster_assignment_map[i]);
	}

	fclose(f);

	f = fopen("centroids.txt", "w");
	if (f == NULL) {
		printf("Error opening file!\n");
		exit(1);
	}

	for(i = 0; i < clusters; i++) {
		for(j = 0; j < dims_test; j++) {
			fprintf(f, "%f ", cluster_centroids[i][j]);
		}
		fprintf(f, "\n");
	}

	fclose(f);

	//REMEMBER TO FREE POINTERS - THERE IS NO GARBAGE COLLECTION! 
	for(k = 0; k < clusters; k++) {
		free(cluster_centroids[k]);
		free(global_centroids[k]);
	}
	free(cluster_centroids);
	free(global_centroids);

	for(i = 0; i < p; i++) {
		for(j = 0; j < clusters; j++) {
			free(local_cluster_centroids[i][j]);
		}
		free(local_cluster_centroids[i]);
	}
	free(local_cluster_centroids);

	for(i = 0; i < p; i++) {
		free(local_cluster_counts[i]);
	}
	free(local_cluster_counts);

	free(cluster_assignment_map);

	return 0;
}
