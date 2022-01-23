#include "../common/graph.h"
#include "../common/alloc.cuh"
#include "../common/wtime.h"
#include "../common/comm.cuh"
#include "../common/fqg.cuh"
#include "../common/mcpy.cuh"
#include <fstream>

template<typename vertex_t, typename index_t, typename depth_t>
void alloc_sim(

        index_t vert_count,
        depth_t* &sa_d_sim,
        vertex_t* &fq_td_1_d_sim,
        vertex_t* &fq_td_1_curr_sz_sim,
        vertex_t* &fq_sz_h_sim,
        vertex_t* &fq_td_2_d_sim,
        vertex_t* &fq_td_2_curr_sz_sim,
        vertex_t* &fq_bu_curr_sz_sim
){

    long gpu_bytes = 0;

    H_ERR(cudaMalloc((void **) &sa_d_sim, sizeof(depth_t) * vert_count));
    H_ERR(cudaMalloc((void **) &fq_td_1_d_sim, sizeof(vertex_t) * vert_count));
    H_ERR(cudaMalloc((void **) &fq_td_1_curr_sz_sim, sizeof(vertex_t)));
    H_ERR(cudaMallocHost((void **) &fq_sz_h_sim, sizeof(vertex_t)));
    H_ERR(cudaMalloc((void **) &fq_td_2_d_sim, sizeof(vertex_t) * vert_count));
    H_ERR(cudaMalloc((void **) &fq_td_2_curr_sz_sim, sizeof(vertex_t)));
    H_ERR(cudaMalloc((void **) &fq_bu_curr_sz_sim, sizeof(vertex_t)));

    gpu_bytes = sizeof(depth_t) * vert_count + sizeof(vertex_t) * (vert_count * 2 + 3);
    std::cout << "Additional GPU mem for simulation: " << gpu_bytes << std::endl;
}

template<typename vertex_t, typename index_t, typename depth_t>
void dealloc_sim(

        depth_t* &sa_d_sim,
        vertex_t* &fq_td_1_d_sim,
        vertex_t* &fq_td_1_curr_sz_sim,
        vertex_t* &fq_sz_h_sim,
        vertex_t* &fq_td_2_d_sim,
        vertex_t* &fq_td_2_curr_sz_sim,
        vertex_t* &fq_bu_curr_sz_sim
){

    cudaFree(sa_d_sim);
    cudaFree(fq_td_1_d_sim);
    cudaFree(fq_td_1_curr_sz_sim);
    cudaFree(fq_sz_h_sim);
    cudaFree(fq_td_2_d_sim);
    cudaFree(fq_td_2_curr_sz_sim);
    cudaFree(fq_bu_curr_sz_sim);
}

template<typename vertex_t, typename index_t, typename depth_t>
__global__ void mcpy_init_sim(

        index_t vert_count,
        depth_t *sa_d,
        depth_t *sa_d_sim,
        vertex_t *fq_td_1_d,
        vertex_t *fq_td_1_curr_sz,
        vertex_t *fq_td_1_d_sim,
        vertex_t *fq_td_1_curr_sz_sim,
        vertex_t *fq_td_2_d,
        vertex_t *fq_td_2_curr_sz,
        vertex_t *fq_td_2_d_sim,
        vertex_t *fq_td_2_curr_sz_sim,
        vertex_t *fq_bu_curr_sz,
        vertex_t *fq_bu_curr_sz_sim
){

    index_t tid_st = threadIdx.x + blockDim.x * blockIdx.x;
    index_t tid;
    const index_t grnt = gridDim.x * blockDim.x;

    tid = tid_st;
    while(tid < vert_count){

        fq_td_1_d_sim[tid] = fq_td_1_d[tid];
        tid += grnt;
    }

    tid = tid_st;
    while(tid < vert_count){

        fq_td_2_d_sim[tid] = fq_td_2_d[tid];
        tid += grnt;
    }

    tid = tid_st;
    while(tid < vert_count){

        sa_d_sim[tid] = sa_d[tid];
        tid += grnt;
    }

    if(tid_st == 0){

        *fq_td_1_curr_sz_sim = *fq_td_1_curr_sz;
        *fq_td_2_curr_sz_sim = *fq_td_2_curr_sz;
        *fq_bu_curr_sz_sim = *fq_bu_curr_sz;
    }
}

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_td(

        depth_t *sa_d,
        const vertex_t * __restrict__ adj_list_d,
        const index_t * __restrict__ offset_d,
        const index_t * __restrict__ adj_deg_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_td_in_d,
        vertex_t *fq_td_in_curr_sz,
        vertex_t *fq_sz_h,
        vertex_t *fq_td_out_d,
        vertex_t *fq_td_out_curr_sz
){

    if(*fq_sz_h < (vertex_t) (par_beta * vert_count)){

        fqg_td_wccao<vertex_t, index_t, depth_t> // warp-cooperative chained atomic operations
        <<<BLKS_NUM_TD_WCCAO, THDS_NUM_TD_WCCAO>>>(

                sa_d,
                adj_list_d,
                offset_d,
                adj_deg_d,
                level,
                fq_td_in_d,
                fq_td_in_curr_sz,
                fq_td_out_d,
                fq_td_out_curr_sz
        );
        cudaDeviceSynchronize();
    }

    else{

        fqg_td_wcsac<vertex_t, index_t, depth_t> // warp-cooperative status array check
        <<<BLKS_NUM_TD_WCSAC, THDS_NUM_TD_WCSAC>>>(

                sa_d,
                adj_list_d,
                offset_d,
                adj_deg_d,
                level,
                fq_td_in_d,
                fq_td_in_curr_sz
        );
        cudaDeviceSynchronize();

        fqg_td_tcfe<vertex_t, index_t, depth_t> // thread-centric frontier enqueue
        <<<BLKS_NUM_TD_TCFE, THDS_NUM_TD_TCFE>>>(

                sa_d,
                vert_count,
                level,
                fq_td_out_d,
                fq_td_out_curr_sz
        );
        cudaDeviceSynchronize();
    }

    H_ERR(cudaMemcpy(fq_sz_h, fq_td_out_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_bu(

        depth_t *sa_d,
        const vertex_t * __restrict__ adj_list_d,
        const index_t * __restrict__ offset_d,
        const index_t * __restrict__ adj_deg_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_sz_h,
        vertex_t *fq_bu_curr_sz
){

    fqg_bu_wcsac<vertex_t, index_t, depth_t>
    <<<BLKS_NUM_BU_WCSA, THDS_NUM_BU_WCSA>>>(

            sa_d,
            adj_list_d,
            offset_d,
            adj_deg_d,
            vert_count,
            level,
            fq_bu_curr_sz
    );
    cudaDeviceSynchronize();

    H_ERR(cudaMemcpy(fq_sz_h, fq_bu_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_rev(

        depth_t *sa_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_sz_h,
        vertex_t *fq_td_in_d,
        vertex_t *fq_td_in_curr_sz
){

    fqg_rev_tcfe<vertex_t, index_t, depth_t> // thread-centric frontier enqueue
    <<<BLKS_NUM_REV_TCFE, THDS_NUM_REV_TCFE>>>(

            sa_d,
            vert_count,
            level,
            fq_td_in_d,
            fq_td_in_curr_sz
    );
    cudaDeviceSynchronize();

    H_ERR(cudaMemcpy(fq_sz_h, fq_td_in_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

template<typename vertex_t, typename index_t, typename depth_t>
void bfs_tdbu(

        depth_t *sa_d,
        const vertex_t * __restrict__ adj_list_d,
        const index_t * __restrict__ offset_d,
        const index_t * __restrict__ adj_deg_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_td_1_d,
        vertex_t *temp_fq_td_d,
        vertex_t *fq_td_1_curr_sz,
        vertex_t *temp_fq_curr_sz,
        vertex_t *fq_sz_h,
        vertex_t *fq_td_2_d,
        vertex_t *fq_td_2_curr_sz,
        vertex_t *fq_bu_curr_sz,
        vertex_t INFTY,
        depth_t *sa_d_sim,
        vertex_t *fq_td_1_d_sim,
        vertex_t *fq_td_1_curr_sz_sim,
        vertex_t *fq_sz_h_sim,
        vertex_t *fq_td_2_d_sim,
        vertex_t *fq_td_2_curr_sz_sim,
        vertex_t *fq_bu_curr_sz_sim,
        std::ofstream &data_train
){

    vertex_t prev_fq_sz = 0;
    vertex_t curr_fq_sz = 0;
    vertex_t unvisited = (vertex_t) vert_count;
    double prev_slope = 0.0;
    double curr_slope = 0.0; // slope (variation)
    double curr_conv = 0.0; // convexity (tendency)
    double curr_proc = 0.0; // processed (progress)
    double remn_proc = 0.0;

    bool fq_swap = true;
    bool reversed = false;
    bool TD_BU = false; // true: bottom-up, false: top-down

    bool fq_swap_sim;
    bool reversed_sim;
    bool TD_BU_sim;

    *fq_sz_h = 1;
    flush_fq<vertex_t, index_t, depth_t>
    <<<1, 1>>>(

            fq_bu_curr_sz
    );
    cudaDeviceSynchronize();

    double t_st, t_end, t_acc, t_avg_td, t_avg_bu, t_st_rev, t_end_rev;

    for(level = 0; ; level++){

        if(level == 0){
            TD_BU_sim = false;
            H_ERR(cudaMemcpy(fq_sz_h_sim, fq_sz_h, sizeof(vertex_t), cudaMemcpyHostToHost));
        }
        else{

            for(int i = 0; i < 2; i++){

                TD_BU_sim = (bool) i;
                t_acc = 0.0;

                for(int j = 0; j < NUM_SIM; j++){

                    t_st_rev = 0;
                    t_end_rev = 0;

                    // 1. Initialize simulation data structures
                    H_ERR(cudaMemcpy(fq_sz_h_sim, fq_sz_h, sizeof(vertex_t), cudaMemcpyHostToHost));
                    fq_swap_sim = fq_swap;

                    if(TD_BU && !TD_BU_sim)
                        reversed_sim = true;
                    else
                        reversed_sim = false;

                    mcpy_init_sim<vertex_t, index_t, depth_t>
                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                            vert_count,
                            sa_d,
                            sa_d_sim,
                            fq_td_1_d,
                            fq_td_1_curr_sz,
                            fq_td_1_d_sim,
                            fq_td_1_curr_sz_sim,
                            fq_td_2_d,
                            fq_td_2_curr_sz,
                            fq_td_2_d_sim,
                            fq_td_2_curr_sz_sim,
                            fq_bu_curr_sz,
                            fq_bu_curr_sz_sim
                    );
                    cudaDeviceSynchronize();

                    // 2. Simulate traversal and evaluate runtime
                    t_st = wtime();

                    if(!TD_BU_sim){

                        if(!fq_swap_sim)
                            fq_swap_sim = true;
                        else
                            fq_swap_sim = false;

                        if(level != 0){

                            if(!reversed_sim){

                                if(!fq_swap_sim){

                                    mcpy_init_fq_td<vertex_t, index_t, depth_t>
                                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                            vert_count,
                                            temp_fq_td_d,
                                            temp_fq_curr_sz,
                                            fq_td_2_d_sim,
                                            fq_td_2_curr_sz_sim,
                                            INFTY
                                    );
                                }

                                else{

                                    if(level == 1){

                                        init_fqg_2<vertex_t, index_t, depth_t>
                                        <<<1, 1>>>(

                                                fq_td_1_d_sim,
                                                fq_td_1_curr_sz_sim,
                                                INFTY
                                        );
                                    }

                                    else{

                                        mcpy_init_fq_td<vertex_t, index_t, depth_t>
                                        <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                                vert_count,
                                                temp_fq_td_d,
                                                temp_fq_curr_sz,
                                                fq_td_1_d_sim,
                                                fq_td_1_curr_sz_sim,
                                                INFTY
                                        );
                                    }
                                }
                                cudaDeviceSynchronize();
                            }

                            else{

                                t_st_rev = wtime();

                                reversed_sim = false;
                                fq_swap_sim = false;

                                mcpy_init_fq_td<vertex_t, index_t, depth_t>
                                <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                        vert_count,
                                        temp_fq_td_d,
                                        temp_fq_curr_sz,
                                        fq_td_2_d_sim,
                                        fq_td_2_curr_sz_sim,
                                        INFTY
                                );

                                mcpy_init_fq_td<vertex_t, index_t, depth_t>
                                <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                        vert_count,
                                        temp_fq_td_d,
                                        temp_fq_curr_sz,
                                        fq_td_1_d_sim,
                                        fq_td_1_curr_sz_sim,
                                        INFTY
                                );
                                cudaDeviceSynchronize();

                                bfs_rev<vertex_t, index_t, depth_t>(

                                        sa_d_sim,
                                        vert_count,
                                        level,
                                        fq_sz_h_sim,
                                        fq_td_1_d_sim,
                                        fq_td_1_curr_sz_sim
                                );
                                cudaDeviceSynchronize();
                                t_end_rev = wtime();
                            }
                        }

                        if(!fq_swap_sim){

                            bfs_td<vertex_t, index_t, depth_t>(

                                    sa_d_sim,
                                    adj_list_d,
                                    offset_d,
                                    adj_deg_d,
                                    vert_count,
                                    level,
                                    fq_td_1_d_sim,
                                    fq_td_1_curr_sz_sim,
                                    fq_sz_h_sim,
                                    fq_td_2_d_sim,
                                    fq_td_2_curr_sz_sim
                            );
                        }

                        else{

                            bfs_td<vertex_t, index_t, depth_t>(

                                    sa_d_sim,
                                    adj_list_d,
                                    offset_d,
                                    adj_deg_d,
                                    vert_count,
                                    level,
                                    fq_td_2_d_sim,
                                    fq_td_2_curr_sz_sim,
                                    fq_sz_h_sim,
                                    fq_td_1_d_sim,
                                    fq_td_1_curr_sz_sim
                            );
                        }

                        cudaDeviceSynchronize();
                    }
                    else{

                        flush_fq<vertex_t, index_t, depth_t>
                        <<<1, 1>>>(

                                fq_bu_curr_sz_sim
                        );
                        cudaDeviceSynchronize();

                        bfs_bu<vertex_t, index_t, depth_t>(

                                sa_d_sim,
                                adj_list_d,
                                offset_d,
                                adj_deg_d,
                                vert_count,
                                level,
                                fq_sz_h_sim,
                                fq_bu_curr_sz_sim
                        );
                        cudaDeviceSynchronize();
                    }

                    t_end = wtime();

                    // 3. Accumulate runtime
                    t_acc += (t_end - t_st) - (t_end_rev - t_st_rev);
                }

                // 4. Assign avg_runtime
                if(!TD_BU_sim)
                    t_avg_td = t_acc;
                else
                    t_avg_bu = t_acc;
            }

            // 5. Select the proper direction by average runtime
            if(t_avg_td <= t_avg_bu)
                TD_BU_sim = false;
            else
                TD_BU_sim = true;
        }

        // 6. Generate train_data
        if(level == 0){
            prev_fq_sz = 0;
            prev_slope = 0.0;
        }
        else{
            prev_fq_sz = curr_fq_sz;
            prev_slope = curr_slope;
        }

        curr_fq_sz = *fq_sz_h;
        curr_slope = ((double) curr_fq_sz - prev_fq_sz) / vert_count;
        curr_conv = curr_slope - prev_slope;

        unvisited -= curr_fq_sz;
        curr_proc = (double) curr_fq_sz / vert_count;
        remn_proc = (double) unvisited / vert_count;

        // Input features
        data_train << avg_deg << ",";
        data_train << prob_high << ",";
        data_train << curr_slope << ",";
        data_train << curr_conv << ",";
        data_train << curr_proc << ",";
        data_train << remn_proc << ",";
        // Label
        data_train << TD_BU_sim << std::endl;
        num_data++;

        // 7. Assign direction to TD_BU
        if(!TD_BU_sim){

            if(TD_BU)
                reversed = true;

            TD_BU = false;
        }

        else
            TD_BU = true;

        // 9. Actual traversal
        if(!TD_BU){

            if(!fq_swap)
                fq_swap = true;
            else
                fq_swap = false;

            if(level != 0){

                if(!reversed){

                    if(!fq_swap){

                        mcpy_init_fq_td<vertex_t, index_t, depth_t>
                        <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                vert_count,
                                temp_fq_td_d,
                                temp_fq_curr_sz,
                                fq_td_2_d,
                                fq_td_2_curr_sz,
                                INFTY
                        );
                    }

                    else{

                        if(level == 1){

                            init_fqg_2<vertex_t, index_t, depth_t>
                            <<<1, 1>>>(

                                    fq_td_1_d,
                                    fq_td_1_curr_sz,
                                    INFTY
                            );
                        }

                        else{

                            mcpy_init_fq_td<vertex_t, index_t, depth_t>
                            <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                    vert_count,
                                    temp_fq_td_d,
                                    temp_fq_curr_sz,
                                    fq_td_1_d,
                                    fq_td_1_curr_sz,
                                    INFTY
                            );
                        }
                    }
                }

                else{

                    reversed = false;
                    fq_swap = false;

                    mcpy_init_fq_td<vertex_t, index_t, depth_t>
                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                            vert_count,
                            temp_fq_td_d,
                            temp_fq_curr_sz,
                            fq_td_2_d,
                            fq_td_2_curr_sz,
                            INFTY
                    );

                    mcpy_init_fq_td<vertex_t, index_t, depth_t>
                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                            vert_count,
                            temp_fq_td_d,
                            temp_fq_curr_sz,
                            fq_td_1_d,
                            fq_td_1_curr_sz,
                            INFTY
                    );
                    cudaDeviceSynchronize();

                    bfs_rev<vertex_t, index_t, depth_t>(

                            sa_d,
                            vert_count,
                            level,
                            fq_sz_h,
                            fq_td_1_d,
                            fq_td_1_curr_sz
                    );
                }
            }

            cudaDeviceSynchronize();

            if(!fq_swap){

                bfs_td<vertex_t, index_t, depth_t>(

                        sa_d,
                        adj_list_d,
                        offset_d,
                        adj_deg_d,
                        vert_count,
                        level,
                        fq_td_1_d,
                        fq_td_1_curr_sz,
                        fq_sz_h,
                        fq_td_2_d,
                        fq_td_2_curr_sz
                );
            }

            else{

                bfs_td<vertex_t, index_t, depth_t>(

                        sa_d,
                        adj_list_d,
                        offset_d,
                        adj_deg_d,
                        vert_count,
                        level,
                        fq_td_2_d,
                        fq_td_2_curr_sz,
                        fq_sz_h,
                        fq_td_1_d,
                        fq_td_1_curr_sz
                );
            }

            cudaDeviceSynchronize();
        }
        else{

            flush_fq<vertex_t, index_t, depth_t>
            <<<1, 1>>>(

                    fq_bu_curr_sz
            );
            cudaDeviceSynchronize();

            bfs_bu<vertex_t, index_t, depth_t>(

                    sa_d,
                    adj_list_d,
                    offset_d,
                    adj_deg_d,
                    vert_count,
                    level,
                    fq_sz_h,
                    fq_bu_curr_sz
            );
            cudaDeviceSynchronize();
        }

        if(*fq_sz_h == 0)
            break;
    }
}

// Function called from CPU
template<typename vertex_t, typename index_t, typename depth_t>
int bfs( // breadth-first search on GPU

        vertex_t *src_list,
        index_t *beg_pos,
        vertex_t *csr,
        index_t vert_count,
        index_t edge_count,
        index_t gpu_id,
        std::ofstream &data_train,
        vertex_t INFTY
){

    srand((unsigned int) wtime());
    int retry = 0;

    cudaSetDevice(gpu_id);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    depth_t *sa_d; // status array on GPU
    depth_t *sa_h; // status array on CPU
    depth_t *temp_sa; // initial state of status array (used for iterative test)
    index_t *adj_deg_d; // the number of neighbors for each vertex
    index_t *adj_deg_h;
    vertex_t *adj_list_d; // adjacent lists
    index_t *offset_d; // offset
    vertex_t *fq_td_1_d; // frontier queue for top-down traversal
    vertex_t *fq_td_1_curr_sz; // used for the top-down queue size
                            // synchronized index of frontier queue for top-down traversal, the size must be 1
    vertex_t *fq_td_2_d;
    vertex_t *fq_td_2_curr_sz;
    vertex_t *temp_fq_td_d;
    vertex_t *temp_fq_curr_sz;
    vertex_t *fq_sz_h;
    vertex_t *fq_bu_curr_sz; // used for the number of vertices examined at each level, the size must be 1

    // sim_data_structures
    depth_t *sa_d_sim;
    vertex_t *fq_td_1_d_sim;
    vertex_t *fq_td_1_curr_sz_sim;
    vertex_t *fq_sz_h_sim;
    vertex_t *fq_td_2_d_sim;
    vertex_t *fq_td_2_curr_sz_sim;
    vertex_t *fq_bu_curr_sz_sim;

    alloc<vertex_t, index_t, depth_t>::
    alloc_mem(

            sa_d,
            sa_h,
            temp_sa,
            adj_list_d,
            adj_deg_d,
            adj_deg_h,
            offset_d,
            beg_pos,
            csr,
            vert_count,
            edge_count,
            fq_td_1_d,
            temp_fq_td_d,
            fq_td_1_curr_sz,
            temp_fq_curr_sz,
            fq_sz_h,
            fq_td_2_d,
            fq_td_2_curr_sz,
            fq_bu_curr_sz
    );

    alloc_sim<vertex_t, index_t, depth_t>(

            vert_count,
            sa_d_sim,
            fq_td_1_d_sim,
            fq_td_1_curr_sz_sim,
            fq_sz_h_sim,
            fq_td_2_d_sim,
            fq_td_2_curr_sz_sim,
            fq_bu_curr_sz_sim
    );

    mcpy_init_temp<vertex_t, index_t, depth_t>
    <<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(

            vert_count,
            temp_fq_td_d,
            temp_fq_curr_sz,
            INFTY
    );
    cudaDeviceSynchronize();

    depth_t level;

    warm_up_gpu<<<BLKS_NUM_INIT, THDS_NUM_INIT>>>();
    cudaDeviceSynchronize();

    ///// iteration starts /////////////////////////////////////////////////////////////////////////////////////////////

    for(index_t i = 0; i < NUM_ITER; i++){
        H_ERR(cudaMemcpy(sa_d, temp_sa, sizeof(depth_t) * vert_count, cudaMemcpyHostToDevice));
        H_ERR(cudaMemcpy(sa_h, temp_sa, sizeof(depth_t) * vert_count, cudaMemcpyHostToHost));

        mcpy_init_fq_td<vertex_t, index_t, depth_t>
        <<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(

                vert_count,
                temp_fq_td_d,
                temp_fq_curr_sz,
                fq_td_1_d,
                fq_td_1_curr_sz,
                INFTY
        );
        cudaDeviceSynchronize();

        mcpy_init_fq_td<vertex_t, index_t, depth_t>
        <<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(

                vert_count,
                temp_fq_td_d,
                temp_fq_curr_sz,
                fq_td_2_d,
                fq_td_2_curr_sz,
                INFTY
        );
        cudaDeviceSynchronize();

        init_fqg<vertex_t, index_t, depth_t>
        <<<1, 1>>>(

                src_list[i],
                sa_d,
                fq_td_1_d,
                fq_td_1_curr_sz
        );
        cudaDeviceSynchronize();

        level = 0;

        if(!retry){

            std::cout << "===========================================================" << std::endl;
            std::cout << "<<Iteration " << i << ">>" << std::endl;
//        std::cout << "Started from " << src_list[i] << std::endl;
        }

        calc_par_opt<vertex_t, index_t>(

                adj_deg_h,
                vert_count,
                edge_count
        );

        bfs_tdbu<vertex_t, index_t, depth_t>(

                sa_d,
                adj_list_d,
                offset_d,
                adj_deg_d,
                vert_count,
                level,
                fq_td_1_d,
                temp_fq_td_d,
                fq_td_1_curr_sz,
                temp_fq_curr_sz,
                fq_sz_h,
                fq_td_2_d,
                fq_td_2_curr_sz,
                fq_bu_curr_sz,
                INFTY,
                sa_d_sim,
                fq_td_1_d_sim,
                fq_td_1_curr_sz_sim,
                fq_sz_h_sim,
                fq_td_2_d_sim,
                fq_td_2_curr_sz_sim,
                fq_bu_curr_sz_sim,
                data_train
        );

        // for validation
        index_t tr_vert = 0;
        index_t tr_edge = 0;

        H_ERR(cudaMemcpy(sa_h, sa_d, sizeof(depth_t) * vert_count, cudaMemcpyDeviceToHost));

        for(index_t j = 0; j < vert_count; j++){
            if(sa_h[j] != UNVISITED){

                tr_vert++;
                tr_edge += adj_deg_h[j];
            }
        }

        // Retry the traversal due to bad source (the input graph is disconnected)
        if(tr_vert < (double) vert_count * 0.5 || tr_edge < (double) edge_count * 0.7){

            src_list[i] = rand() % vert_count;
            i--;
            retry++;
            continue;
        }
        retry = 0;

        std::cout << "Started from " << src_list[i] << std::endl;
        std::cout << "The number of traversed vertices: " << tr_vert << std::endl;
        std::cout << "The number of traversed edges: " << tr_edge << std::endl;
        std::cout << "Depth: " << level << std::endl;
    }

    std::cout << "===========================================================" << std::endl;
    std::cout << "Newly generated data: " << num_data <<std::endl;

    ///// iteration ends ///////////////////////////////////////////////////////////////////////////////////////////////

    alloc<vertex_t, index_t, depth_t>::
    dealloc_mem(

            sa_d,
            sa_h,
            temp_sa,
            adj_list_d,
            adj_deg_d,
            adj_deg_h,
            offset_d,
            fq_td_1_d,
            temp_fq_td_d,
            fq_td_1_curr_sz,
            temp_fq_curr_sz,
            fq_sz_h,
            fq_td_2_d,
            fq_td_2_curr_sz,
            fq_bu_curr_sz
    );

    dealloc_sim<vertex_t, index_t, depth_t>(

            sa_d_sim,
            fq_td_1_d_sim,
            fq_td_1_curr_sz_sim,
            fq_sz_h_sim,
            fq_td_2_d_sim,
            fq_td_2_curr_sz_sim,
            fq_bu_curr_sz_sim
    );

    std::cout << "GPU BFS finished" << std::endl;

    return 0;
}
