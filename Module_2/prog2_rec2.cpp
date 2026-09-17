// Online C++ compiler to run C++ program online
#include <iostream>
#include <cstring>

#define SIZE 100

using namespace std;

class STACK {
    private: 
        char num[SIZE];
        int top;
    public: 
        STACK();
        STACK(int a);
        char push(char);
        char pop();
        int isEmpty();
        int isFull();
};

STACK::STACK(){
    top = -1;
}

STACK::STACK(int a){
    top = a;
}

char STACK::push(char a){
    // check if stack is full
    if(isFull()){return 0;}
    // Increment top
    ++top;
    num[top] = a;
    return a;
}

char STACK::pop(){
    // store whatever is deleted
    char temp;
    if(isEmpty()){return 0;}
    temp = num[top];
    --top;
    return temp;
}

int STACK::isEmpty(){
    if(top == -1){
        return 1;
    }
    else{
        return 0;
    }
}

int STACK::isFull(){
    if(top == SIZE - 1){
        return 1;
    }
    else{
        return 0;    
    }
}

bool isValid(char* s){

    // get length of string
    int length = strlen(s);
    int len = 0; 

    // could also do it this way
    while(s[len] != '/0'){
        len++;
    }

    // initialize stack
    STACK st(-1);
    // set default value of empty stack to -1;
     
    for(int i = 0; i < length; i++){
        char c = s[i];

        if(c == '('){
            st.push(')');
        }
        else if(c == '{'){
            st.push('}');
        }
        else if(c == '['){
            st.push(']');
        }
        else if(c == ')' || c == '}' || c == ']'){
            if(st.isEmpty() || st.pop() != c){
                return false;
            }
        }
    }

    // this should be only true if everything is matched and popped
    return st.isEmpty();
    
}

int main() {
    // Testcases
    char test1[] = "()";
    char test2[] = "()[]{}";
    char test3[] = "(}";
    char test4[] = "({})";
    char test5[] = "([)]";

    // 1 is correct, 0 is incorrect
    cout << "Test 1: " << isValid(test1) << endl;
    cout << "Test 2: " << isValid(test2) << endl; 
    cout << "Test 3: " << isValid(test3) << endl; 
    cout << "Test 4: " << isValid(test4) << endl;
    cout << "Test 5: " << isValid(test5) << endl;

    return 0;
}