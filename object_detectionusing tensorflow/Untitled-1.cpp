//
// Created by SOHAM on 01-09-2026.
//
#include <iostream>
using namespace std;

class Number {
    int x;
public:
    Number(int a) {
        x = a;
    }
    Number operator++(int) {
        Number temp = *this;
        x++;
        return temp;
    }
    void display() {
        cout << x << endl;
    }
};
int main() {
    Number n(10);
    n++;
    n.display();
    return 0;
}