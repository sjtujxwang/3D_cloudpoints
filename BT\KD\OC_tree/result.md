result:
  
   kdtree:
          knn_grade=kdtree_knn_time/k_brute_time=4.49768239697983
          rnn_grade=kdtree_rnn_time/r_brute_time=0.004339701132727173
          
         remarks:
   k_brute_time: Time spent on violent k-search
   kdtree_knn_time: Time spent on  knn-search with kdtree
   r_brute_time:Time spent on violent r-search
   kdtree_rnn_time:Time spent on  rnn-search with kdtree


   octree:
          OCTree_knn_grade=octtree_knn_time/k_brute_time=3.6258395014687466
          OCTree_rnn_grade=octtree_rnn_time/r_brute_time=0.043036583296383275
          OCTree_rnn_fast_grade=octtree_rnn_fast_time/r_brute_time=0.04056820981065147
          
         remarks:
   k_brute_time: Time spent on violent k-search
   r_brute_time: Time spent on violent r-search
   octtree_knn_time: Time spent on  knn-search with octree
   octtree_rnn_time: Time spent on  rnn-search with octree
   octtree_rnn_fast_time:Time spent on improved  rnn-search with octree

