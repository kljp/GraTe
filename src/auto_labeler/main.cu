#include "../common/graph.h"
#include "bfs_al.cuh"
#include <sstream>
#include <fstream>
#include <climits>

template <typename vertex_t, typename index_t, typename depth_t>
void process_graph(

        std::string file_beg_pos,
        std::string file_adj_list,
        std::string file_data_train,
        vertex_t INFTY
)
{

    const vertex_t gpu_id = 0;

    std::ofstream data_train;
    data_train.open(file_data_train.c_str(), std::ios::out | std::ios::app);
    if(!data_train.is_open()){
        std::cout << "File open error." << std::endl;
        exit(-1);
    }

    graph<vertex_t, index_t, double, vertex_t, index_t, double> *ginst
            = new graph<vertex_t, index_t, double, vertex_t, index_t, double>(file_beg_pos.c_str(), file_adj_list.c_str(), NULL);
    srand((unsigned int) wtime());

    vertex_t *src_list = new vertex_t[NUM_ITER];
    vertex_t src;
    for(int i = 0; i < NUM_ITER; i++){

        src = rand() % ginst->vert_count;

        if(ginst->beg_pos[src + 1] - ginst->beg_pos[src] > 0)
            src_list[i] = src;
        else
            i--;
    }

    bfs<vertex_t, index_t, depth_t>(

            src_list,
            ginst->beg_pos,
            ginst->csr,
            ginst->vert_count,
            ginst->edge_count,
            gpu_id,
            data_train,
            INFTY
    );

    data_train.close();

    delete[] src_list;
    delete ginst;
}

int main(int argc, char **argv){

    if(argc < 6){

        std::cout
                << "Required argument:\n"
                << "\t--csr : beg_pos and adj_list of input graph (e.g., --csr com-Orkut.mtx_beg_pos.bin com-Orkut.mtx_adj_list.bin)\n"
                << "\t--data : filename of train data\n"
                << "Optional argument:\n"
                << "\t--verylarge : set data type of vertices and edges to ' unsigned long long' to handle very large input graph (e.g., com-Friendster), default='unsigned int'\n"
                << "\t--verbose : print breakdown of frontier processing techniques\n"
                << std::endl;

        exit(-1);
    }

    std::string file_beg_pos;
    std::string file_adj_list;
    std::string file_data_train;
    bool is_verylarge = false;
    bool is_verbose = false;
    bool is_checked_input = false;
    bool is_checked_data = false;
    bool is_checked_verylarge = false;
    bool is_checked_verbose = false;

    for(int i = 1; i < argc; i++){
        if(!strcmp(argv[i], "--csr") && i != argc - 1 && i != argc - 2){
            if(!is_checked_input){
                file_beg_pos = std::string(argv[i + 1]);
                file_adj_list = std::string(argv[i + 2]);
                is_checked_input = true;
            }
        }
        else if(!strcmp(argv[i], "--data") && i != argc - 1){
            if(!is_checked_data){
                file_data_train = std::string(argv[i + 1]);
                is_checked_data = true;
            }
        }
        else if(!strcmp(argv[i], "--verylarge")){
            if(!is_checked_verylarge){
                is_verylarge = true;
                is_checked_verylarge = true;
            }
        }
        else if(!strcmp(argv[i], "--verbose")){
            if(!is_checked_verbose){
                is_verbose = true;
                is_checked_verbose = true;
            }
        }
    }

    if(is_verbose)
        verbose = true;
    else
        verbose = false;

    if(is_verylarge){
        std::cout << "Data type='unsigned long long'" << std::endl;
        process_graph<unsigned long long, unsigned long long, unsigned int>(
                file_beg_pos,
                file_adj_list,
                file_data_train,
                ULLONG_MAX
        );
    }
    else{
        std::cout << "Data type='unsigned int'" << std::endl;
        process_graph<unsigned int, unsigned int, unsigned int>(
                file_beg_pos,
                file_adj_list,
                file_data_train,
                UINT_MAX
        );
    }

    return 0;
}
