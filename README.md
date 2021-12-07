# GraTe
## BFS Direction-Optimizer utilizing tendency of workload on GPUs
---
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
TBD

---
Code specification
-----
__GraTe implementation:__
TBD

__CSR Generator provided by https://github.com/kljp/vCSR/:__
- vcsr.cpp: generate CSR
    - Compile: make
    - Execute: ./vcsr --input \<\*.mtx\> \[option1\] \[option2\] \[option3\] \[option4\]
      - \[option1\]: --virtual \<max\_degree\> \(not available for FADO\)
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
Contact
-----
If you have any questions about this project, contact me by one of the followings:
- slashxp@naver.com
- kljp@ajou.ac.kr