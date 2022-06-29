# SURF
## BFS Direction-Optimizer utilizing workload state of frontiers on GPUs.
---
SURF is abbreviation of 'switching direction upon recent workload state'.

This project aims to support a high performance breadth-first graph traversal on GPUs.

---
Tested operating system
-----
Ubuntu \[18.04.5, 20.04.2\] LTS

---
Tested software
-----
g++ \[7.5.0, 9.3.0\], CUDA \[11.2, 11.5\]

---
Tested hardware
-----
GTX970, RTX3080

---
Compile
-----
make

---
Execute
-----
./surf --csr \<\*_beg_pos.bin\> \<\*_adj_list.bin\> \[option1\] \[option2\]
- \[option1\]: --verylarge
    - set data type of vertices and edges to 'unsigned long long', default='unsigned int'
- \[option2\]: --verbose
    - print breakdown of frontier processing techniques

---
Code specification
-----
__SURF implementation:__
- main.cu: load a graph as an input
- bfs.cuh: traverse the graph
- model.h: initialize a trained MLP model and predict a label of direction
- fqg.cuh: implementation of push and pull phases
- mcpy.cuh: functions for initializing data structures
- alloc.cuh: memory allocation for data structures
- comm.cuh: global variables and functions shared by all files

__Auto labeler:__
- bfs_al.cu: generate train data as auto-labeled records
    - Compile: make
    - Execute: ./aula --csr \<\*_beg_pos.bin\> \<\*_adj_list.bin\> --data \<train_data\> \[option1\]
      - \[option1\]: --verylarge
        - set data type of vertices and edges to 'unsigned long long', default='unsigned int'

__CSR Generator provided by https://github.com/kljp/vCSR/:__
- vcsr.cpp: generate CSR
    - Compile: make
    - Execute: ./vcsr --input \<\*.mtx\> \[option1\] \[option2\] \[option3\] \[option4\]
      - \[option1\]: --virtual \<max\_degree\> \(not available for SURF\)
        - set maximum degree of a vertex to \<max\_degree\>
      - \[option2\]: --undirected
        - add reverse edges
      - \[option3\]: --sorted
        - sort intra-neighbor lists
      - \[option4\]: --verylarge
        - set data type of vertices and edges to 'unsigned long long', default='unsigned int'
    - Graph source: https://sparse.tamu.edu/
    - Please make sure that the format of input graph should be Matrix Market.

__Headers Provided by https://github.com/iHeartGraph/Enterprise/:__
- graph.h: graph data structure
- graph.hpp: implementation of graph data structure
- wtime.h: get current time for measuring the consumed time

---
Publication
-----
Daegun Yoon, Sangyoon Oh, **SURF: Direction-Optimizing Breadth-First Search Using Workload State on GPUs**, *Sensors*, Jun. 2022. [**\[Paper\]**](https://www.mdpi.com/1424-8220/22/13/4899)

---
Contact
-----
If you have any questions about this project, contact me by one of the followings:
- slashxp@naver.com
- kljp@ajou.ac.kr