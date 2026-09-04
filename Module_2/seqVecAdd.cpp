const int N = 8;
float A[N] = {1,2,3,4,5,6,7,8};
float B[N] = {8,7,6,5,4,3,2,1};
float C[N];

void vector_add(){
    for(int i = 0; i < N; i++){
        C[i] = A[i] + B[i];
    }
}

int main(){
    vector_add();
}