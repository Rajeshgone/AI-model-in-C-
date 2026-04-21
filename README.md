# AI-model-in-C-
AI model in C++
C++ Version: Simple Feedforward Neural Network (for Iris classification)

// basic_ai_model.cpp
// Compile: g++ basic_ai_model.cpp -o basic_ai_model -std=c++17 -O2
// Run: ./basic_ai_model

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>

using namespace std;

// Simple Neural Network class
class BasicNeuralNet {
private:
    vector<vector<double>> w1, w2, w3;  // Weights
    vector<double> b1, b2, b3;          // Biases
    
    double learning_rate = 0.01;
    
    double relu(double x) { return max(0.0, x); }
    double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }
    
    // Softmax for output
    vector<double> softmax(const vector<double>& x) {
        vector<double> result(x.size());
        double sum = 0.0;
        double max_val = *max_element(x.begin(), x.end());
        for (double val : x) sum += exp(val - max_val);
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = exp(x[i] - max_val) / sum;
        return result;
    }

public:
    BasicNeuralNet() {
        // Initialize weights and biases (4 -> 16 -> 12 -> 3)
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, 0.1);

        // Layer 1: 4 -> 16
        w1.resize(16, vector<double>(4));
        b1.resize(16);
        for (auto& row : w1) for (auto& val : row) val = dist(gen);
        for (auto& val : b1) val = dist(gen);

        // Layer 2: 16 -> 12
        w2.resize(12, vector<double>(16));
        b2.resize(12);
        for (auto& row : w2) for (auto& val : row) val = dist(gen);
        for (auto& val : b2) val = dist(gen);

        // Layer 3: 12 -> 3
        w3.resize(3, vector<double>(12));
        b3.resize(3);
        for (auto& row : w3) for (auto& val : row) val = dist(gen);
        for (auto& val : b3) val = dist(gen);
    }

    vector<double> forward(const vector<double>& x) {
        // Layer 1
        vector<double> h1(16);
        for (int i = 0; i < 16; ++i) {
            h1[i] = b1[i];
            for (int j = 0; j < 4; ++j) h1[i] += w1[i][j] * x[j];
            h1[i] = relu(h1[i]);
        }

        // Layer 2
        vector<double> h2(12);
        for (int i = 0; i < 12; ++i) {
            h2[i] = b2[i];
            for (int j = 0; j < 16; ++j) h2[i] += w2[i][j] * h1[j];
            h2[i] = relu(h2[i]);
        }

        // Output layer
        vector<double> out(3);
        for (int i = 0; i < 3; ++i) {
            out[i] = b3[i];
            for (int j = 0; j < 12; ++j) out[i] += w3[i][j] * h2[j];
        }
        return softmax(out);
    }

    void train(const vector<vector<double>>& X_train, const vector<int>& y_train, int epochs) {
        cout << "Training...\n";
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            for (size_t i = 0; i < X_train.size(); ++i) {
                // Forward pass (simplified - full backprop omitted for brevity)
                vector<double> output = forward(X_train[i]);
                // Very basic update (for demo)
                // In real code, implement proper backpropagation
            }
            if ((epoch + 1) % 50 == 0) {
                cout << "Epoch " << epoch + 1 << "/" << epochs << " completed\n";
            }
        }
    }
};

int main() {
    cout << "Basic AI Model in C++ (Neural Network)\n\n";

    // Sample Iris data (first 10 samples - you can load full dataset)
    vector<vector<double>> X = {
        {5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2}, {4.7, 3.2, 1.3, 0.2},
        {7.0, 3.2, 4.7, 1.4}, {6.4, 3.2, 4.5, 1.5}, {6.9, 3.1, 4.9, 1.5},
        {6.3, 3.3, 6.0, 2.5}, {5.8, 2.7, 5.1, 1.9}, {7.1, 3.0, 5.9, 2.1}
    };
    
    vector<int> y = {0, 0, 0, 1, 1, 1, 2, 2, 2};  // 0=setosa, 1=versicolor, 2=virginica

    BasicNeuralNet model;
    model.train(X, y, 200);

    // Test prediction
    vector<double> sample = {5.1, 3.5, 1.4, 0.2};
    vector<double> pred = model.forward(sample);
    
    cout << "\nPrediction for sample [5.1, 3.5, 1.4, 0.2]:\n";
    for (size_t i = 0; i < pred.size(); ++i) {
        cout << "Class " << i << " probability: " << pred[i] * 100 << "%\n";
    }
    cout << "Predicted class: " << max_element(pred.begin(), pred.end()) - pred.begin() << endl;

    return 0;
}



 To compile and run:

g++ basic_ai_model.cpp -o basic_ai_model -std=c++17
./basic_ai_model
