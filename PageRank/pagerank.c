
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "pr_graph.h"

#define _POSIX_C_SOURCE 199309L
#include <time.h>
/* OSX timer includes */
#ifdef __MACH__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif



/** latest
* @brief Compute the PageRank (PR) of a graph.
*
* @param graph The graph.
* @param damping Damping factor (or, 1-restart). 0.85 is typical.
* @param max_iterations The maximium number of iterations to perform.
*
* @return A vector of PR values.
*/

double * pagerank(
    pr_graph const * const graph,
    double const damping,
    int const max_iterations, pr_int const nvtxs, int rank, pr_int l);

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


int main(int argc, char * * argv) {

  int npes,rank;
  MPI_Status status;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&npes);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  //Master process
  if(rank == 0) {

	  if(argc == 1) {
	    fprintf(stderr, "usage: %s <graph> [output file]\n", *argv);
	    return EXIT_FAILURE;
	  }
	
	  char * ifname = argv[1];
	  char * ofname = NULL;
	  if(argc > 2) {
	    ofname = argv[2];
	  }
	
	  pr_graph * graph = pr_graph_load(ifname);
	  if(!graph) {
	    return EXIT_FAILURE;
	  }

          //pr_int const nvtxs = graph->nvtxs;
          //pr_int const * const restrict xadj = graph->xadj;
          //pr_int const * const restrict nbrs = graph->nbrs;

	  pr_int local_nvtxs = graph->nvtxs/npes;
  	  pr_int rem_nvtxs = graph->nvtxs%npes;

	  pr_graph * graph_l = malloc(sizeof(*graph_l));

	  graph_l->nvtxs = (rem_nvtxs>0) ? local_nvtxs+1 : local_nvtxs;
	  graph_l->nedges = graph->xadj[graph_l->nvtxs] - graph->xadj[0];
	  graph_l->xadj = malloc((graph_l->nvtxs+1)*sizeof(*graph_l->xadj));
	  graph_l->nbrs = malloc(graph_l->nedges*sizeof(*graph_l->nbrs));
	  for(pr_int i=0; i<(graph_l->nvtxs+1); i++)
		graph_l->xadj[i] = graph->xadj[i];
	  for(pr_int i=0; i<graph_l->nedges; i++)
		graph_l->nbrs[i] = graph->nbrs[i];

	  pr_int len_adj = graph_l->nvtxs;
	  pr_int len_nbrs = graph_l->nedges;

	  // splitting and sending
	  for(int send = 1; send < npes; send++) {
		pr_graph * graph_l = malloc(sizeof(*graph_l));

		graph_l->nvtxs = (rem_nvtxs>send) ? local_nvtxs+1 : local_nvtxs;
		graph_l->nedges = graph->xadj[graph_l->nvtxs+len_adj] - graph->xadj[len_adj];
		graph_l->xadj = malloc((graph_l->nvtxs+1)*sizeof(*graph_l->xadj));
		graph_l->nbrs = malloc(graph_l->nedges*sizeof(*graph_l->nbrs));
		for(pr_int i=0; i<(graph_l->nvtxs+1); i++)
			graph_l->xadj[i] = graph->xadj[i+len_adj];
		for(pr_int i=0; i<graph_l->nedges; i++)
			graph_l->nbrs[i] = graph->nbrs[i+len_nbrs];

		pr_int index = graph_l->xadj[0];
		for(pr_int i=0; i<(graph_l->nvtxs+1); i++)
			graph_l->xadj[i] = graph_l->xadj[i]-index;

		MPI_Send(&graph->nvtxs, 1, MPI_UNSIGNED_LONG, send, 1, MPI_COMM_WORLD);
		MPI_Send(&graph_l->nvtxs, 1, MPI_UNSIGNED_LONG, send, 1, MPI_COMM_WORLD);
		MPI_Send(&graph_l->nedges, 1, MPI_UNSIGNED_LONG, send, 1, MPI_COMM_WORLD);
		MPI_Send(graph_l->xadj, graph_l->nvtxs+1, MPI_UNSIGNED_LONG, send, 1, MPI_COMM_WORLD);
		MPI_Send(graph_l->nbrs, graph_l->nedges, MPI_UNSIGNED_LONG, send, 1, MPI_COMM_WORLD);
		MPI_Send(&len_adj, 1, MPI_UNSIGNED_LONG, send, 1, MPI_COMM_WORLD);

		len_adj = len_adj + graph_l->nvtxs;
		len_nbrs = len_nbrs + graph_l->nedges;
	  }
	
	  double * PR = pagerank(graph_l, 0.85, 100, graph->nvtxs, rank, 0);
	
	  /* write pagerank values */
	  if(ofname) {
	    FILE * fout = fopen(ofname, "w");
	    if(!fout) {
	      fprintf(stderr, "ERROR: could not open '%s' for writing.\n", ofname);
	      return EXIT_FAILURE;
	    }
	    for(pr_int v=0; v < graph->nvtxs; ++v) {
	      fprintf(fout, "%0.3e\n", PR[v]);
	    }
	    fclose(fout);
	  }

/*	  // printing local PR values iteratively from each MPI process

	  if(ofname) {
	    FILE * fout = fopen(ofname, "w");
	    if(!fout) {
	      fprintf(stderr, "ERROR: could not open '%s' for writing.\n", ofname);
	      return EXIT_FAILURE;
	    }

	    for(pr_int v=0; v < (rem_nvtxs>0)?graph_l->nvtxs+1:graph_l->nvtxs; ++v) {
	      fprintf(fout, "%0.3e\n", PR[v]);
	    }

	    for(int recv = 1; recv < npes; recv++) {
		if (rem_nvtxs>recv)
		    MPI_Recv(PR, graph_l->nvtxs+1, MPI_DOUBLE, recv, 1, MPI_COMM_WORLD, &status);
		else
		    MPI_Recv(PR, graph_l->nvtxs, MPI_DOUBLE, recv, 1, MPI_COMM_WORLD, &status);
	    	for(pr_int v=0; v < (rem_nvtxs>recv)?graph_l->nvtxs+1:graph_l->nvtxs; ++v) {
	    	    fprintf(fout, "%0.3e\n", PR[v]);
	    	}
	    }
	    fclose(fout);
	  } 	*/
	
	  free(PR);

	  MPI_Finalize();
	
	  return EXIT_SUCCESS;

  } //Master process - end

  //Worker process
  if(rank > 0) {
	pr_int nvtxs, l;	
	pr_graph * graph_l = malloc(sizeof(*graph_l));

	MPI_Recv(&nvtxs, 1, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&graph_l->nvtxs, 1, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&graph_l->nedges, 1, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, &status);
	graph_l->xadj = malloc((graph_l->nvtxs+1)*sizeof(*graph_l->xadj));
	graph_l->nbrs = malloc(graph_l->nedges*sizeof(*graph_l->nbrs));
	MPI_Recv(graph_l->xadj, graph_l->nvtxs+1, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(graph_l->nbrs, graph_l->nedges, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, &status);
	MPI_Recv(&l, 1, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, &status);
	
	double * PR = pagerank(graph_l, 0.85, 100, nvtxs, rank, l);
//	double * PR = pagerank(graph_l, 0.85, 100, nvtxs, rank, nvtxs/npes, nvtxs%npes, npes);	//calling memory-scalable pagerank method

//	MPI_Send(PR, graph_l->nvtxs, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);

	free(PR);

	MPI_Finalize();
  } // Worker process - end

}

double * pagerank(
    pr_graph const * const graph,
    double const damping,
    int const max_iterations, pr_int const nvtxs, int rank, pr_int l)
{
  /* grab graph structures to save typing */
  pr_int const local_nvtxs = graph->nvtxs;
  pr_int const * const restrict local_xadj = graph->xadj;
  pr_int const * const restrict local_nbrs = graph->nbrs;

  /* Initialize pageranks to be a probability distribution. */
  double * PR = malloc(nvtxs * sizeof(*PR));
  for(pr_int v=0; v < nvtxs; ++v) {
    PR[v] = 1. / (double) nvtxs;
  }

  /* Probability of restart */
  double const restart = (1 - damping) / (double) nvtxs;

  /* Convergence tolerance. */
  double const tol = 1e-9;

  double * PR_accum = malloc(nvtxs * sizeof(*PR));
  double * PR_accum_l = malloc(nvtxs * sizeof(*PR));

  double start_time = monotonic_seconds();

  int i;
  for(i=0; i < max_iterations; ++i) {

    for(pr_int v=0; v < nvtxs; ++v) {
      PR_accum_l[v] = 0.;
    }

    /* Each vertex pushes PR contribution to all outgoing links */
    for(pr_int v=0; v < local_nvtxs; ++v) {
      double const num_links = (double)(local_xadj[v+1] - local_xadj[v]);
      double const pushing_val = PR[l+v] / num_links;

      for(pr_int e=local_xadj[v]; e < local_xadj[v+1]; ++e) {
        PR_accum_l[local_nbrs[e]] += pushing_val;
      }
    }

    MPI_Allreduce(PR_accum_l, PR_accum, nvtxs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* Finalize new PR values */
    double norm_changed = 0.;
    for(pr_int v=0; v < nvtxs; ++v) {
      double const old = PR[v];
      PR[v] = restart + (damping * PR_accum[v]);

      norm_changed += (PR[v] - old) * (PR[v] - old);
    }
    norm_changed = sqrt(norm_changed);

    if(i > 1 && norm_changed < tol) {
      break;
    }
  }

  double time_taken = monotonic_seconds() - start_time;
  i = (i==max_iterations)?i:i+1;
  time_taken = time_taken/i;

  if(rank == 0)
	printf("Number of iterations: %d, Average time: %.3f seconds\n",i,time_taken);

  free(PR_accum_l);
  free(PR_accum);

  return PR;
}

// memory-scalable local pagerank calculation
/*
double * pagerank(
    pr_graph const * const graph,
    double const damping,
    int const max_iterations, pr_int const nvtxs, int rank, pr_int local_nodes, pr_int rem_nodes, int npes)
{
  pr_int const local_nvtxs = graph->nvtxs;
  pr_int const * const restrict local_xadj = graph->xadj;
  pr_int const * const restrict local_nbrs = graph->nbrs;

  double * PR = malloc(local_nvtxs * sizeof(*PR));
  for(pr_int v=0; v < local_nvtxs; ++v) {
    PR[v] = 1. / (double) nvtxs;
  }

  double const restart = (1 - damping) / (double) nvtxs;

  double const tol = 1e-9;

  int * local_edge_count = malloc(npes * sizeof(*local_edge_count));
  for(int i=0; i<npes; i++) {
	local_edge_count[i] = 0;
  }
  int * edge_count = malloc(npes * sizeof(*edge_count));
  for(int i=0; i<npes; i++) {
	edge_count[i] = 0;
  }

  for(pr_int v=0; v < local_nvtxs; ++v) {
    for(pr_int e=local_xadj[v]; e < local_xadj[v+1]; ++e) {
	if(rem_nodes == 0)
	  local_edge_count[local_nbrs[e]/local_nodes]++;
	else {
	  if(local_nbrs[e] < (local_nodes + 1)*rem_nodes)
	    local_edge_count[local_nbrs[e]/(local_nodes+1)]++;
	  else
	    local_edge_count[(local_nbrs[e]-rem_nodes)/local_nodes]++;
	}
    }
  }
  
  MPI_Alltoall(local_edge_count, npes, MPI_INT, edge_count, npes, MPI_INT, MPI_COMM_WORLD);

  int local_edge_count_sum = 0, edge_count_sum = 0;
  int * local_vtxs_value = malloc(npes * sizeof(*local_vtxs_value));
  int * vtxs_value = malloc(npes * sizeof(*vtxs_value));

  for (pr_int v = 0; v < npes; v++) {
    local_vtxs_value[v] = local_edge_count_sum;
    local_edge_count_sum = local_edge_count_sum + local_edge_count[v];
  }

  for (pr_int v = 0; v < npes; v++) {
    vtxs_value[v] = edge_count_sum;
    edge_count_sum = edge_count_sum + edge_count[v];
  }

  int * local_vtxs_index = malloc(local_edge_count_sum * sizeof(*local_vtxs_index));
  int * vtxs_index = malloc(edge_count_sum * sizeof(*vtxs_index));

  int c = 0;
  for (pr_int v = 0; v < nvtxs; ++v)
  {
    for (pr_int e = xadj[v]; e < xadj[v + 1]; ++e)
    {
      local_vtxs_index[c] = nbrs[e];
      c++;
    }
  }

  MPI_Alltoallv(local_vtxs_index, local_edge_count, local_vtxs_value, MPI_INT, vtxs_index, edge_count, vtxs_value, MPI_INT, MPI_COMM_WORLD);

  double * local_pagerank = malloc(local_edge_count_sum * sizeof(*local_pagerank));
  double * pagerank = malloc(edge_count_sum * sizeof(*pagerank));

  double * PR_accum = malloc(local_nvtxs * sizeof(*PR_accum));

  double start_time = monotonic_seconds();

  int i;
  for(i=0; i < max_iterations; ++i) {

    for(pr_int v=0; v < local_nvtxs; ++v) {
      PR_accum[v] = 0.;
    }

    for(pr_int v=0; v < local_nvtxs; ++v) {
      double const num_links = (double)(local_xadj[v+1] - local_xadj[v]);
      double const pushing_val = PR[v] / num_links;

      for(pr_int e=local_xadj[v]; e < local_xadj[v+1]; ++e) {
        local_pagerank[local_nbrs[e]] += pushing_val;
      }
    }

    MPI_Alltoallv(local_pagerank, local_edge_count, local_vtxs_value, MPI_DOUBLE, pagerank, edge_count, vtxs_value, MPI_DOUBLE, MPI_COMM_WORLD);

    for(pr_int v=0; v < local_nvtxs; ++v) {
        PR_accum[vtxs_index[v]] += pagerank[v];
    }

    double norm_changed = 0.;
    double total_norm_changed = 0.;
    for(pr_int v=0; v < local_nvtxs; ++v) {
      double const old = PR[v];
      PR[v] = restart + (damping * PR_accum[v]);

      norm_changed += (PR[v] - old) * (PR[v] - old);
    }

    MPI_Allreduce(&norm_changed, &total_norm_changed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    total_norm_changed = sqrt(total_norm_changed);

    if(i > 1 && total_norm_changed < tol) {
      break;
    }
  }

  double time_taken = monotonic_seconds() - start_time;
  i = (i==max_iterations)?i:i+1;
  time_taken = time_taken/i;

  if(rank == 0)
	printf("Number of iterations: %d, Average time: %.3f seconds\n",i,time_taken);

  free(PR_accum);

  return PR;
} */
