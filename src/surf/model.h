#ifndef __H_MODEL__
#define __H_MODEL__

#include <cmath>

double weight_input_hidden[6][6];
double bias_hidden[6];
double weight_hidden_output[6];
double bias_output;

inline double relu(double input){

    if(input > 0)
        return input;
    else
        return 0;
}

inline double sigmoid(double input){

    return 1 / (1 + std::exp(input * -1));
}

inline bool predict(const double *neuron_input){

    bool label;
    double neuron_hidden[6];
    neuron_hidden[0] = relu(neuron_input[0] * weight_input_hidden[0][0]
                            + neuron_input[1] * weight_input_hidden[1][0]
                            + neuron_input[2] * weight_input_hidden[2][0]
                            + neuron_input[3] * weight_input_hidden[3][0]
                            + neuron_input[4] * weight_input_hidden[4][0]
                            + neuron_input[5] * weight_input_hidden[5][0]
                            + bias_hidden[0]);
    neuron_hidden[1] = relu(neuron_input[0] * weight_input_hidden[0][1]
                            + neuron_input[1] * weight_input_hidden[1][1]
                            + neuron_input[2] * weight_input_hidden[2][1]
                            + neuron_input[3] * weight_input_hidden[3][1]
                            + neuron_input[4] * weight_input_hidden[4][1]
                            + neuron_input[5] * weight_input_hidden[5][1]
                            + bias_hidden[1]);
    neuron_hidden[2] = relu(neuron_input[0] * weight_input_hidden[0][2]
                            + neuron_input[1] * weight_input_hidden[1][2]
                            + neuron_input[2] * weight_input_hidden[2][2]
                            + neuron_input[3] * weight_input_hidden[3][2]
                            + neuron_input[4] * weight_input_hidden[4][2]
                            + neuron_input[5] * weight_input_hidden[5][2]
                            + bias_hidden[2]);
    neuron_hidden[3] = relu(neuron_input[0] * weight_input_hidden[0][3]
                            + neuron_input[1] * weight_input_hidden[1][3]
                            + neuron_input[2] * weight_input_hidden[2][3]
                            + neuron_input[3] * weight_input_hidden[3][3]
                            + neuron_input[4] * weight_input_hidden[4][3]
                            + neuron_input[5] * weight_input_hidden[5][3]
                            + bias_hidden[3]);
    neuron_hidden[4] = relu(neuron_input[0] * weight_input_hidden[0][4]
                            + neuron_input[1] * weight_input_hidden[1][4]
                            + neuron_input[2] * weight_input_hidden[2][4]
                            + neuron_input[3] * weight_input_hidden[3][4]
                            + neuron_input[4] * weight_input_hidden[4][4]
                            + neuron_input[5] * weight_input_hidden[5][4]
                            + bias_hidden[4]);
    neuron_hidden[5] = relu(neuron_input[0] * weight_input_hidden[0][5]
                            + neuron_input[1] * weight_input_hidden[1][5]
                            + neuron_input[2] * weight_input_hidden[2][5]
                            + neuron_input[3] * weight_input_hidden[3][5]
                            + neuron_input[4] * weight_input_hidden[4][5]
                            + neuron_input[5] * weight_input_hidden[5][5]
                            + bias_hidden[5]);

    double neuron_output = neuron_hidden[0] * weight_hidden_output[0]
                           + neuron_hidden[1] * weight_hidden_output[1]
                           + neuron_hidden[2] * weight_hidden_output[2]
                           + neuron_hidden[3] * weight_hidden_output[3]
                           + neuron_hidden[4] * weight_hidden_output[4]
                           + neuron_hidden[5] * weight_hidden_output[5]
                           + bias_output;

    neuron_output = sigmoid(neuron_output);
    if(neuron_output < 0.5)
        label = false;
    else
        label = true;

    return label;
}

void init_model(){

    weight_input_hidden[0][0] = -2.303028013557195663e-03;
    weight_input_hidden[0][1] = 4.044337198138237000e-02;
    weight_input_hidden[0][2] = -4.462619423866271973e-01;
    weight_input_hidden[0][3] = -4.787955880165100098e-01;
    weight_input_hidden[0][4] = -6.824228763580322266e-01;
    weight_input_hidden[0][5] = -2.554011642932891846e-01;

    weight_input_hidden[1][0] = 2.062148094177246094e+00;
    weight_input_hidden[1][1] = 1.009470748901367188e+01;
    weight_input_hidden[1][2] = -1.551159024238586426e-01;
    weight_input_hidden[1][3] = 2.363148331642150879e-01;
    weight_input_hidden[1][4] = -4.725298881530761719e-01;
    weight_input_hidden[1][5] = 9.805077314376831055e-02;

    weight_input_hidden[2][0] = -1.033092117309570312e+01;
    weight_input_hidden[2][1] = -1.216687965393066406e+01;
    weight_input_hidden[2][2] = 4.356395602226257324e-01;
    weight_input_hidden[2][3] = 1.415770053863525391e-01;
    weight_input_hidden[2][4] = 5.835133194923400879e-01;
    weight_input_hidden[2][5] = -2.011589705944061279e-01;

    weight_input_hidden[3][0] = 1.322073340415954590e+00;
    weight_input_hidden[3][1] = 6.962181568145751953e+00;
    weight_input_hidden[3][2] = -4.583228826522827148e-01;
    weight_input_hidden[3][3] = 2.352800369262695312e-01;
    weight_input_hidden[3][4] = 4.106130003929138184e-01;
    weight_input_hidden[3][5] = -3.548008203506469727e-01;

    weight_input_hidden[4][0] = 7.143101501464843750e+01;
    weight_input_hidden[4][1] = 2.645717811584472656e+01;
    weight_input_hidden[4][2] = 7.732397317886352539e-02;
    weight_input_hidden[4][3] = -1.423445940017700195e-01;
    weight_input_hidden[4][4] = -5.740886330604553223e-01;
    weight_input_hidden[4][5] = -2.131928130984306335e-02;

    weight_input_hidden[5][0] = 6.917166233062744141e+00;
    weight_input_hidden[5][1] = -6.894137573242187500e+01;
    weight_input_hidden[5][2] = -5.916026830673217773e-01;
    weight_input_hidden[5][3] = 4.553381800651550293e-01;
    weight_input_hidden[5][4] = -2.932451367378234863e-01;
    weight_input_hidden[5][5] = 2.833680249750614166e-02;


    bias_hidden[0] = -1.921773552894592285e-01;
    bias_hidden[1] = -2.072150468826293945e+00;
    bias_hidden[2] = 0.000000000000000000e+00;
    bias_hidden[3] = 0.000000000000000000e+00;
    bias_hidden[4] = 0.000000000000000000e+00;
    bias_hidden[5] = -3.121882677078247070e-02;


    weight_hidden_output[0] = 1.954106390476226807e-01;
    weight_hidden_output[1] = 7.156560420989990234e-01;
    weight_hidden_output[2] = 5.420844554901123047e-01;
    weight_hidden_output[3] = -2.381692528724670410e-01;
    weight_hidden_output[4] = -8.286997079849243164e-01;
    weight_hidden_output[5] = 1.751225255429744720e-02;

    bias_output = -3.250087976455688477e+00;
}

#endif //__H_MODEL__
