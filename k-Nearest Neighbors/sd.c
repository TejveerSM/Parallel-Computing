/*!
\file  main.c
\brief This file is the entry point for paragon's various components
 
\date   Started 11/27/09
\author George
\version\verbatim $Id: omp_main.c 9585 2011-03-18 16:51:51Z karypis $ \endverbatim
*/


#include "simdocs.h"
#include <omp.h>


/*************************************************************************/
/*! This is the entry point for finding simlar patents */
/**************************************************************************/
int main(int argc, char *argv[])
{
  params_t params;
  int rc = EXIT_SUCCESS;

  cmdline_parse(&params, argc, argv);

  printf("********************************************************************************\n");
  printf("sd (%d.%d.%d) Copyright 2011, GK.\n", VER_MAJOR, VER_MINOR, VER_SUBMINOR);
  printf("nthreads=%d, nnbrs=%d, minsim=%.2f\n", params.nthreads, params.nnbrs, params.minsim);

  gk_clearwctimer(params.timer_global);
  gk_clearwctimer(params.timer_1);
  gk_clearwctimer(params.timer_2);
  gk_clearwctimer(params.timer_3);
  gk_clearwctimer(params.timer_4);

  gk_startwctimer(params.timer_global);

  ComputeNeighbors(&params);

  gk_stopwctimer(params.timer_global);

  printf("    wclock: %.2lfs\n", gk_getwctimer(params.timer_global));
  printf("    timer1: %.2lfs\n", gk_getwctimer(params.timer_1));
  printf("    timer2: %.2lfs\n", gk_getwctimer(params.timer_2));
  printf("    timer3: %.2lfs\n", gk_getwctimer(params.timer_3));
  printf("    timer4: %.2lfs\n", gk_getwctimer(params.timer_4));
  printf("********************************************************************************\n");

  exit(rc);
}


/*************************************************************************/
/*! Reads and computes the neighbors of each document */
/**************************************************************************/
void ComputeNeighbors(params_t *params)
{
  int i, j, nhits;
  gk_csr_t *mat;
  int32_t *marker;
  gk_fkv_t *hits, *cand;
  FILE *fpout;

  printf("Reading data for %s...\n", params->infstem);

  mat = gk_csr_Read(params->infstem, GK_CSR_FMT_CSR, 1, 0);

  printf("#docs: %d, #nnz: %d.\n", mat->nrows, mat->rowptr[mat->nrows]);

  /* compact the column-space of the matrices */
  gk_csr_CompactColumns(mat);

  /* perform auxiliary pre-computations based on similarity */
  gk_csr_ComputeSquaredNorms(mat, GK_CSR_ROW);

  /* create the inverted index */
  gk_csr_CreateIndex(mat, GK_CSR_COL);

  /* create the output file */
  fpout = (params->outfile ? gk_fopen(params->outfile, "w", "ComputeNeighbors: fpout") : NULL);


/*
  gk_startwctimer(params->timer_1);
  #pragma omp parallel shared(mat,params,fpout) private(hits,marker,cand,nhits,i,j) num_threads(params->nthreads)
{
  hits   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: hits");
  marker = gk_i32smalloc(mat->nrows, -1, "ComputeNeighbors: marker");
  cand   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: cand");

  for(i=0;i<10;i++) {
    hits_local[i] = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: hits");
    marker_local[i] = gk_i32smalloc(mat->nrows, -1, "ComputeNeighbors: marker");
    cand_local[i] = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: cand");
  }
  
  #pragma omp for
  for(int a=0;a<mat->nrows;a=a+(mat->nrows/10)) {
        gk_csr_t *A = gk_csr_ExtractSubmatrix(mat,a,mat->nrows/10);
            for (i=0; i<A->nrows; i++) {
              if (params->verbosity > 0)
                printf("Working on query %7d\n", i);

	      #pragma omp for */
              /* find the neighbors of the ith document
              for(int b=0;b<mat->nrows;b=b+(mat->nrows/10)) {
                gk_csr_t *B = gk_csr_ExtractSubmatrix(mat,b,mat->nrows/10);
              	nhits = gk_csr_GetSimilarRows(B,
                 	  B->rowptr[i+1]-B->rowptr[i],
                 	  B->rowind+B->rowptr[i],
                 	  B->rowval+B->rowptr[i],
                 	  GK_CSR_JAC, params->nnbrs, params->minsim, hits_local[b],
                 	  marker_local[b], cand_local[b]);
              }
              temp_hits = gk_fkvmalloc(mat->nrows*10, "ComputeNeighbors: hits")
	      for(int k=0;k<10;k++) {
		temp_hits = temp_hits->add(hits_local[k]); //ToDo code - adding up all local_hits to temp_hits
	      }
	      hits = top 'nnbrs' of temp_hits; //ToDo code - selecting the nearest values */

              /* write the results in the file
              if (fpout) {
                for (j=0; j<nhits; j++)
                fprintf(fpout, "%8d %8zd %.3f\n", k+i, hits[j].val, hits[j].key);
              }
              
              gk_csr_Free(&B);
            }
        gk_csr_Free(&A);
   }
}
  gk_stopwctimer(params->timer_1);
*/



  /* find the best neighbors for each query document */
  gk_startwctimer(params->timer_1);
  #pragma omp parallel shared(mat,params,fpout) private(hits,marker,cand,nhits,i,j) num_threads(params->nthreads)
  {
  hits   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: hits");
  marker = gk_i32smalloc(mat->nrows, -1, "ComputeNeighbors: marker");
  cand   = gk_fkvmalloc(mat->nrows, "ComputeNeighbors: cand");

  #pragma omp for
  for (i=0; i<mat->nrows; i++) {
    if (params->verbosity > 0)
      printf("Working on query %7d\n", i);

    /* find the neighbors of the ith document */ 
    nhits = gk_csr_GetSimilarRows(mat, 
                 mat->rowptr[i+1]-mat->rowptr[i], 
                 mat->rowind+mat->rowptr[i], 
                 mat->rowval+mat->rowptr[i], 
                 GK_CSR_JAC, params->nnbrs, params->minsim, hits, 
                 marker, cand);

    /* write the results in the file */
    if (fpout) {
      for (j=0; j<nhits; j++) 
        fprintf(fpout, "%8d %8zd %.3f\n", i, hits[j].val, hits[j].key);
    }
  }
  gk_free((void **)&hits, &marker, &cand, LTERM);
  }
  gk_stopwctimer(params->timer_1);


  /* cleanup and exit */
  if (fpout) gk_fclose(fpout);

  gk_csr_Free(&mat);

  return;
}

