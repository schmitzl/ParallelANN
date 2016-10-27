/* Use MPI */
#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

/* --------------------------------------------------------------------------------------*/
/* STRUCTS */

/*
 *  Node of a mulitlayer perceptron with the weights its input is multiplied with.
 */
struct node {
    double delta;
};

/*
 * Layer of a multilayer perceptron. It contains the inputs and their amount as well as
 * the nodes in this layer and their amount.
 * Assumption is that every node in this layer has the same inputs.
 */
struct layer {
    int sizeInputs;
    int sizeN;
    double *inputs;
    double **weights;
    struct node *nodes;
};

/*
 * Multilayer perceptron with its amount of layers L, its layer as well as
 * its final output and the amount of those outputs.
 */
struct mlp {
    int sizeL;
    struct layer *layers;
    int sizeOutput;
    double *outputs;
    double *targets;
};


/* --------------------------------------------------------------------------------------*/
/* FUNCTION DECLARATION */

/*
 *  Creates a local representation of the complete network for each processor. The amounts
 *  of nodes per layer for the processor p in question depends on the rank p and the total
 *  amount of processors P.
 *
 *  sizeL:          amount of layers in the multilayer perceptron; this includes the input
 *                  and the output layer
 *  *sizesN:        total amount N of nodes in each layer
 *  *sizeInputs:    size of the inputs in each layer
 *  p:              rank of processor
 *  P:              total amount of processors
 *
 *  returns: struct mlp with the specified features
 */
struct mlp createMLP( int sizeL, int *sizesN, int *sizeInputs, int p, int P );


/*  Calculates activation by using the transfer function.
 *
 *  sizeInputs: total amount of input parameters
 *  inputs:     inputs of node
 *  weights:    weights of node
 *
 *  returns: activation of node n
 */
double calculateActiviation(int sizeInputs, double* inputs, double *weights);

/*  Computes the responsible nodes for processor p in a certain layer
 *
 *  p:      rank of the processor
 *  P:      total amount of processors
 *  sizeN:  size of nodes in layer
 *
 *  returns: amount of nodes processor p is responsible for
 */
int I(int p, int P, int sizeN);

/*
 *  Transfer function.
 *
 *  x: should be the summed and weighed input for a neuron
 *
 *  returns: the activation
 */
double calculateTransfer(double x);

/*
 *  Inverse transfer function.
 *
 *  x: function parameter
 *
 *  returns: the result of the inverse of the transfer function applied to x
 */
double calculateTransferInverse(double x);

/*
 *  Calculates the deltas for the last layer.
 *
 *  outputs:            outputs of the last layer
 *  targets:            target labels for the input given to the network
 *  globalNodeIndex:    global index of node in last layer for which delta should
 *                      be calculated
 *
 *  returns: the result of the inverse of the transfer function applied to x
 */
double calculateDeltaOfLastLayer(double *outputs, double *targets, int globalNodeIndex);

/*
 *  Calculates the delta for the node with the global index j and the activation h_j.
 *
 *  h_j:            activation of node with global index j. Result of applying the
 *                  transfer function on the weighted input for node j.
 *  j:              global index of node
 *  deltasO:        deltas that have been computed for the nodes in layer l+1
 *  sizeDeltasO:    size of the vector deltasO
 *  v:              weight matrix for the next layer. Contains all the weights of
 *                  every node in the next layer.
 *
 */
double calculateDelta(double h_j, int j, double* deltasO, int sizeDeltasO, double** v);

/*
 *  Calculates the global index given the local index of a node.
 *
 *  p:          rank of processor
 *  P:          amount of processors
 *  i:          local index of node
 *  sizeN:      amount of nodes in current layer
 *
 *  returns:    the global index of a node with the local index i
 */
int muInverse(int p, int P, int i, int sizeN);

/* --------------------------------------------------------------------------------------*/
/* FUNCTIONS */
int main(int argc, char **argv)
{
    /* initialize communication */
    int P, p;
    MPI_Status status;
    int tag = 1;
    double start, end;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &p);
    
    
    /* set sizes for an example multilayer perceptron */
    int sizeL = 5; /* amount of layers */
    int uniformNodes = 64000;
    int *sizeN = (int[5]){16, uniformNodes,uniformNodes, uniformNodes, 2}; /* amount of nodes per layer */
    int *sizeInputs = malloc(sizeof(int)*sizeL); /* amount of input to each layer */
    
    /* set input size according to amount of nodes in the pervious layer */
    int i;
    for(i = 0; i < sizeL; i++) {
        if(i > 0)
            sizeInputs[i] = sizeN[i-1];
        else
            sizeInputs[i] = 0;  /* layer 0 (input layer) has no input and no weights */
    }
    
    double *h_0 = (double[16]){1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; /* input for first layer */
    
    /* create multilayer preceptron with specified features (sizes) from above */
    struct mlp m_Mlp = createMLP(sizeL, sizeN, sizeInputs, p, P);
    
    /* start time measurement */
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
        
        /* -------------------------------------- FORWARD PASS -------------------------------------- */
        /* computation of forward pass */
        /* required: input h_0 and multilayer perceptron m_Mlp */
        
        int l, n, r; /* run variables for layers (l), nodes (n) and processors (r) */
        
        for (l = 1; l < m_Mlp.sizeL; l++) { /* compute activations of nodes in every layer */
            /* -- LAYER -- */
            
            /* I_p: amount of nodes in layer l that processor p is responsible for */
            int I_p = I(p, P, m_Mlp.layers[l].sizeN);
            
            /* store activations (size: I_p) for nodes in layer l that processor p computes in myActivations */
            double *myActivations = (double *) malloc(sizeof(double)*I_p);
            
            /* BEGIN PARALLEL ACTIVATION CALCULATION */
            for (n = 0; n < I_p; n++) { /* compute activation for every node in this layer l */
                /* -- NODE -- */
                int globalNodeIndex = muInverse(p, P, n, m_Mlp.layers[l].sizeN);
                if (l > 1) {
                    myActivations[n] = calculateActiviation( m_Mlp.layers[l].sizeInputs,m_Mlp.layers[l].inputs, m_Mlp.layers[l].weights[globalNodeIndex]);
                }
                else {
                    myActivations[n] = calculateActiviation( m_Mlp.layers[l].sizeInputs, h_0, m_Mlp.layers[l].weights[globalNodeIndex]);
                }
            }
            /* END PARALLEL ACTIVATION CALCULATION */
            
            int upper;
            if(P < m_Mlp.layers[l].sizeN)
                upper = P;
            else
                upper = m_Mlp.layers[l].sizeN;
            
            /* BEGIN COMMUNICATION */
            for (r = 0; r < upper; r++) { /* iterate only through processors that are responsible for the calculations in a node */
                
                int indexStartNode =  muInverse(r, P, 0, m_Mlp.layers[l].sizeN); /* calculate the position of the first element of the input vector for the next layer where processor r computes the activation */
                
                /* find amount of nodes the processor r is responsible for to determine size of buffer */
                int I_r = I(r, P, m_Mlp.layers[l].sizeN);
                int sizeBuf =  I_r;
                
                double *buf = (double *) malloc(sizeof(double)*sizeBuf); /* create buffer for input elements calculated by processor r */
                
                /* if current processor p == r copy the calculated activations into the buffer */
                int b;
                if(p == r) {
                    for(b = 0; b < sizeBuf; b++) {
                        buf[b] = myActivations[b];
                    }
                }
                
                /* broadcast the buffer with the input elements of processor j to all other processors */
                MPI_Bcast(buf, sizeBuf, MPI_DOUBLE, r, MPI_COMM_WORLD);
                
                if(l < m_Mlp.sizeL-1) {
                    for(b = 0; b < sizeBuf; b++) {
                        m_Mlp.layers[l+1].inputs[indexStartNode+b] = buf[b];
                    }
                } else {
                    /* if the activation for the last layer is computed the results should be stored
                     in a special output vector instead as inputs for the next layer */
                    for(b = 0; b < sizeBuf; b++) {
                        m_Mlp.outputs[indexStartNode+b] = buf[b];
                    }
                }
            }
            /* END COMMUNICATION */
        }
        
        /* -------------------------------------- BACKWARD PASS -------------------------------------- */
        
        int sizeDeltas =  m_Mlp.layers[sizeL-1].sizeN;
        double eta = 0.01; /* set learning rate */
        double* deltas;
        
        for (l = m_Mlp.sizeL-1; l > 0; l--) {
            
            /* BEGIN PARALLEL DELTA CALCULATION */
            
            int I_p = I(p, P, m_Mlp.layers[l].sizeN);
            double* myDeltas = (double *) malloc(sizeof(double)*I_p);
            
            /* calculate deltas for the last layer */
            if( l == m_Mlp.sizeL-1) {
                for (n = 0; n < I_p; n++) {
                    /* -- NODE -- */
                    int globalNodeIndex = muInverse(p, P, n, m_Mlp.layers[l].sizeN);
                    double delta = calculateDeltaOfLastLayer(m_Mlp.outputs, m_Mlp.targets, globalNodeIndex);
                    myDeltas[n] = delta;
                }
            } else { /* calculate deltas for other layers */
                for(n = 0; n < I_p; n++) {
                    int globalNodeIndex = muInverse(p, P, n, m_Mlp.layers[l].sizeN);
                    myDeltas[n] = calculateDelta(m_Mlp.layers[l+1].inputs[globalNodeIndex], globalNodeIndex, deltas, m_Mlp.layers[l+1].sizeN, m_Mlp.layers[l+1].weights);
                }
            }
            
            sizeDeltas = m_Mlp.layers[l].sizeN;
            deltas = (double *) malloc(sizeof(double)*sizeDeltas);
            
            /* END PARALLEL DELTA CALCULATION */
            
            /* BEGIN COMMUNICATION */
            int upper;
            if(P < m_Mlp.layers[l].sizeN)
                upper = P;
            else
                upper = m_Mlp.layers[l].sizeN;
            
            /* get all deltas that have been computed for this layer */
            for (r = 0; r < upper; r++) { /* iterate only through processors that are responsible for the calculations in a node */
                
                /* find amount of nodes that the previous processor is responsible for; this is necessary
                 to determine which positions of the input for the next layer will be calculated by processor r */
                int indexStartNode = muInverse(r, P, 0, m_Mlp.layers[l].sizeN);
                
                /* find amount of nodes the processor r is responsible for to determine size of buffer */
                int I_r = I(r, P, m_Mlp.layers[l].sizeN);
                int sizeBuf =  I_r;
                
                double *deltaBuf = (double *) malloc(sizeof(double)*sizeBuf); /* create buffer for deltas calculated by processor r */
                
                /* if current processor p == r copy the calculated activations into the buffer */
                int b;
                if(p == r) {
                    for(b = 0; b < sizeBuf; b++) {
                        deltaBuf[b] = myDeltas[b];
                    }
                }
                
                /* broadcast the buffer with the input elements of processor j to all other processors */
                MPI_Bcast(deltaBuf, sizeBuf, MPI_DOUBLE, r, MPI_COMM_WORLD);
                
                for(b = 0; b < sizeBuf; b++) {
                    deltas[indexStartNode+b] = deltaBuf[b];
                }
            }
            /* END COMMUNICATION */
            
            /* now weight update can be performed by every processor for all the weights */
            int w_l, w_n;
            
            for (w_l = 0; w_l < m_Mlp.layers[l].sizeN; w_l++) {
                for (w_n = 0; w_n < m_Mlp.layers[l].sizeInputs; w_n++) {
                    m_Mlp.layers[l].weights[w_l][w_n] -= eta * m_Mlp.layers[l].inputs[w_n] * deltas[w_l];
                }
            }
            
            
        }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    MPI_Finalize();
    
    if (p == 0) { /* use time from procesor with rank 0 */
        printf("Runtime = %f\n", end-start);
    }
    
    exit(0);
}

struct mlp createMLP( int sizeL, int *sizesN, int *sizeInputs, int p, int P )
{
    
    /* create layers */
    struct layer *layers = (struct layer *) malloc(sizeof(struct layer)*sizeL);
    
    srandom(P); /* intialize random number generator */
    
    int l, n, w_l, w_n, t; /* run variables for layers (l), nodes (n), weights (w_l, w_n) and targets (t) */
    
    for(l = 0; l < sizeL; l++) { /* create layers */
        
        /* allocate space for the inputs of each layer; the inputs are later calculated in the training phase */
        double *inputs =  (double *) malloc(sizeof(double)*sizeInputs[l]);
        
        /* allocate space for the local amount of nodes on this processor */
        int I_p = I(p, P, sizesN[l]);
        struct node *nodes = (struct node *) malloc(sizeof(struct node)*I_p);
        
        /* allocate space for weights per layer */
        double **weights = malloc(sizeof(double)*sizesN[l]);
        
        for(n = 0; n < I_p; n++) { /* create nodes */
            struct node m_Node; /* create node with weights */
            
            nodes[n] = m_Node;
        }
        
        for(w_l = 0; w_l < sizesN[l]; w_l++) {
            /* allocate space for weights per node depending on the size of the inputs */
            weights[w_l] = malloc(sizeof(double)*sizeInputs[l]);
            
            /* generate small random weights in range [-1; 1] */
            for(w_n = 0; w_n < sizeInputs[l]; w_n++) {
                weights[w_l][w_n] = ((double) random())/(RAND_MAX)*2.0 - 1.0;
            }
        }
        
        struct layer m_Layer = {sizeInputs[l], sizesN[l], inputs, weights, nodes}; /* create layer with inputs and nodes */
        layers[l] = m_Layer;
    }
    
    /* allocate space for the final output of the network */
    double *outputs = (double *) malloc(sizeof(double)*sizesN[sizeL-1]);
    double *targets = (double *) malloc(sizeof(double)*sizesN[sizeL-1]);
    for(t = 0; t < sizesN[sizeL-1]; t++) /* initialize targets with some values between 0 and 1 */
        targets[t] = pow(-1, t); /* use constant values [0, 1] here for test example */
            
        /* create the local neural network part */
            struct mlp m_Mlp= {.sizeL = sizeL, .layers = layers, .sizeOutput = sizesN[sizeL-1], .outputs = outputs, .targets = targets};
            return m_Mlp;
}

double calculateActiviation(int sizeInputs, double* inputs, double *weights)
{
    double sumInputs = 0;
    int i;
    for (i = 0; i < sizeInputs; i++) {
        sumInputs += inputs[i]*weights[i];
    }
    
    return calculateTransfer(sumInputs);
}

double calculateTransfer(double x)
{
    double e = exp(1.0);
    return ( 2.0 / ( 1.0 + pow(e, -x) ) - 1.0 );
}

double calculateTransferInverse(double x)
{
    return ( ( (1.0 + x) * (1.0 - x) ) / 2.0 );
}


int I(int p, int P, int sizeN)
{
    int I = (sizeN + P - p - 1) / P;
    
    return I;
    
}

double calculateDeltaOfLastLayer(double *outputs, double *targets, int globalNodeIndex)
{
    double delta = (outputs[globalNodeIndex]-targets[globalNodeIndex]) * calculateTransferInverse(outputs[globalNodeIndex]);
    
    return delta;
}

double calculateDelta(double h_j, int j, double* deltasO, int sizeDeltasO, double** v )
{
    int sum = 0;
    int k;
    for (k = 0; k < sizeDeltasO; k++) {
        sum += deltasO[k] * v[k][j];
    }
    
    double delta_j = sum * calculateTransferInverse(h_j);
    return delta_j;
    return 0;
}


int muInverse(int p, int P, int i, int sizeN)
{
    int K = sizeN / P;
    int R = sizeN % P;
    
    return (p*K + MIN(p, R) + i);
}