#include <thread>
#include <vector>
using namespace std;

const int N = 8, NUM_THREADS = 4;
float A[N] = {1, 2, 3, 4, 5, 6, 7, 8};
float B[N] = {8, 7, 6, 5, 4, 3, 2, 1};
float C[N];

void vector_add(int tid){           // Each CPU thread runs this
    int chunk = N / NUM_THREADS;    // Divide work into chunks
    int start = tid * chunk;        // Thread's first element
    int end = start + chunk;        // Thread's last element

    for(int i = start; i < end; i++){ // Process assigned elements
        C[i] = A[i] + B[i];
    }
}

int main(){
    vector<thread> threads;

    for(int t = 0; t < NUM_THREADS; t++){ // Create 4 CPU Threads
        threads.emplace_back(vector_add, t);
    }

    for(auto& th : threads)
        th.join();
}



