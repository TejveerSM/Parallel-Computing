#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#define _POSIX_C_SOURCE 199309L
#include <time.h>
/* OSX timer includes */
#ifdef __MACH__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

int data_points, dim;
int clusters, threads;
double **data, **centroids;
int *assignment;
int converged;

typedef struct
{
int start;
int value;
} thread_struct;

static inline double monotonic_seconds()
{
	#ifdef __MACH__
	static mach_timebase_info_data_t info;
	static double seconds_per_unit;
	if(seconds_per_unit == 0)
	{
		mach_timebase_info(&info);
		seconds_per_unit = (info.numer / info.denom) / 1e9;
	}
	return seconds_per_unit * mach_absolute_time();
	#else
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
	#endif
}

static void print_time(double const seconds)
{
	printf("k-means clustering time: %0.04fs\n", seconds);
}

double eucli_dist(double *data_row, double *centroid_row)
{
	double d = 0.0;
	for(int i=0; i<dim; i++)
		d = d + ((data_row[i]-centroid_row[i]) * (data_row[i]-centroid_row[i]));
	return d;
}

void *cluster_assignments(void *p)
{
	int i,j;
	double dist, min_dist;
	int min_idx;
	thread_struct *par = (thread_struct *)p;
	for(i=par->start; i<(par->start + par->value); i++)
	{
		min_dist = eucli_dist(data[i],centroids[0]);
		min_idx = 0;
		for(j=1; j<clusters; j++)
		{
			dist = eucli_dist(data[i],centroids[j]);
			if(dist<min_dist)
			{
				min_dist = dist;
				min_idx = j;
			}		
		}
		if(assignment[i] != min_idx)
		{
			assignment[i] = min_idx;
			converged = 0;
		}		
			
	}
	return (void *)0;
}

void *update_centroids(void *p)
{
	int i,j,k;
	double temp_array[dim];
	int count;
	thread_struct *par = (thread_struct *)p;
	for(i=par->start; i<(par->start + par->value); i++)
	{
		count = 0;
		for(k=0; k<dim; k++)
			temp_array[k] = 0;
		for(j=0; j<data_points; j++)
		{
			if(assignment[j] == i)
			{
				for(k=0; k<dim; k++)
					temp_array[k] += data[j][k];
				count++;
			}
		}
		for(k=0; k<dim; k++)
			centroids[i][k] = temp_array[k]/(double)count;
	}
	return (void *)0;
}

void update_thread_centroids()
{
	int temp = clusters;
	int per_thread = temp/threads;
	int curr=0, i;
	thread_struct *p;
	int rem = temp%threads;

	pthread_t tid[threads];

	for(i=0; i<(threads-rem); i++)
	{	
		p = (thread_struct *)malloc(sizeof(thread_struct));
		p->start = curr;
		p->value = per_thread;
		pthread_create(&tid[i], NULL, update_centroids, (void *)p);
		curr += per_thread;
	}
	per_thread++;
	for(i=(threads-rem); i<threads; i++)
	{
		p = (thread_struct *)malloc(sizeof(thread_struct));
		p->start = curr;
		p->value = per_thread;
		pthread_create(&tid[i], NULL, update_centroids, (void *)p);
		curr += per_thread;
	}
	for(i=0; i<threads; i++)
	{
		pthread_join(tid[i],NULL);
	}
}

void cluster_thread_assignments()
{
	int temp = data_points;
	int per_thread = temp/threads;
	int curr=0, i;
	thread_struct *p;
	int rem = temp%threads;

	pthread_t tid[threads];

	for(i=0; i<(threads-rem); i++)
	{	
		p = (thread_struct *)malloc(sizeof(thread_struct));
		p->start = curr;
		p->value = per_thread;
		pthread_create(&tid[i], NULL, cluster_assignments, (void *)p);
		curr += per_thread;
	}
	per_thread++;
	for(i=(threads-rem); i<threads; i++)
	{
		p = (thread_struct *)malloc(sizeof(thread_struct));
		p->start = curr;
		p->value = per_thread;
		pthread_create(&tid[i], NULL, cluster_assignments, (void *)p);
		curr += per_thread;
	}
	for(i=0; i<threads; i++)
	{
		pthread_join(tid[i],NULL);
	}
}
     
int main(int argc, char **argv)
{
	int i,j;
	clusters = atoi(argv[2]);
	threads = atoi(argv[3]);

	// reading data from file
	FILE *fp;
	fp = fopen(argv[1],"r");
	fscanf(fp,"%d",&data_points);
	fscanf(fp,"%d",&dim);
	
	data = (double **)malloc(data_points * sizeof(double *));
	for(i=0; i<data_points; i++) 
		data[i] = (double *)malloc(dim * sizeof(double));

	centroids = (double **)malloc(clusters * sizeof(double *));
	for(i=0; i<clusters; i++) 
		centroids[i] = (double *)malloc(dim * sizeof(double));

	assignment = (int *)malloc(data_points * sizeof(int));

	for(i=0; i<data_points; i++)
	{
		assignment[i] = 0;
		for(j=0; j<dim; j++)
			fscanf(fp,"%lf",&data[i][j]);
	}
	fclose(fp);

	double start = monotonic_seconds();

	// initial centroids
	for(i=0; i<clusters; i++)
		for(j=0; j<dim; j++)
			centroids[i][j] = data[i][j];

	cluster_thread_assignments();

	int iteration = 0;
	converged = 0;

	while(!converged && iteration<20)
	{
		converged = 1;

		update_thread_centroids();
		
		cluster_thread_assignments();

		iteration++;
	}

	double end = monotonic_seconds() - start;

	print_time(end);

	// writing to output files
	FILE *fpw;
	fpw = fopen("clusters.txt","w");
	for(i=0; i<data_points; i++)
		fprintf(fpw,"%d\n",assignment[i]);
	fclose(fpw);

	FILE *fpw2;
	fpw2 = fopen("centroids.txt","w");
	for(i=0; i<clusters; i++)
	{
		for(j=0; j<dim; j++)
			fprintf(fpw2,"%lf ",centroids[i][j]);
		fprintf(fpw2,"\n");
	}
	fclose(fpw2);

	return 0;
}
