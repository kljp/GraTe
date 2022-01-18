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

    weight_input_hidden[0][0] = -1.117256641387939453e+00;
    weight_input_hidden[0][1] = -1.658650040626525879e+00;
    weight_input_hidden[0][2] = 2.487315796315670013e-02;
    weight_input_hidden[0][3] = -1.259038448333740234e+00;
    weight_input_hidden[0][4] = -3.185095498338341713e-03;
    weight_input_hidden[0][5] = -5.573866143822669983e-02;

    weight_input_hidden[1][0] = -2.531924247741699219e+00;
    weight_input_hidden[1][1] = 5.253770828247070312e+00;
    weight_input_hidden[1][2] = 6.235598564147949219e+00;
    weight_input_hidden[1][3] = 1.798212127685546875e+02;
    weight_input_hidden[1][4] = -3.473807525634765625e+01;
    weight_input_hidden[1][5] = 2.896977043151855469e+01;

    weight_input_hidden[2][0] = -5.582130551338195801e-01;
    weight_input_hidden[2][1] = -2.775382614135742188e+01;
    weight_input_hidden[2][2] = -2.580341491699218750e+02;
    weight_input_hidden[2][3] = -3.489633277058601379e-02;
    weight_input_hidden[2][4] = -4.019395262002944946e-02;
    weight_input_hidden[2][5] = -3.902198076248168945e-01;

    weight_input_hidden[3][0] = 2.706017345190048218e-03;
    weight_input_hidden[3][1] = 2.391710281372070312e+01;
    weight_input_hidden[3][2] = 2.072065509855747223e-02;
    weight_input_hidden[3][3] = 2.006917260587215424e-02;
    weight_input_hidden[3][4] = 2.253496274352073669e-02;
    weight_input_hidden[3][5] = 2.382176816463470459e-01;

    weight_input_hidden[4][0] = -9.260103106498718262e-01;
    weight_input_hidden[4][1] = -4.923905944824218750e+01;
    weight_input_hidden[4][2] = -3.456246643066406250e+02;
    weight_input_hidden[4][3] = -3.961721801757812500e+02;
    weight_input_hidden[4][4] = 1.501199722290039062e+01;
    weight_input_hidden[4][5] = -2.033208274841308594e+01;

    weight_input_hidden[5][0] = -2.269636154174804688e+00;
    weight_input_hidden[5][1] = 2.715357065200805664e+00;
    weight_input_hidden[5][2] = -5.181360244750976562e-01;
    weight_input_hidden[5][3] = 9.664932250976562500e+00;
    weight_input_hidden[5][4] = -3.988323211669921875e+01;
    weight_input_hidden[5][5] = -1.257931423187255859e+01;


    bias_hidden[0] = -2.284490108489990234e+00;
    bias_hidden[1] = 3.628284931182861328e+00;
    bias_hidden[2] = 1.127725481986999512e+00;
    bias_hidden[3] = 1.811111602783203125e+02;
    bias_hidden[4] = -2.854812240600585938e+01;
    bias_hidden[5] = 1.841198158264160156e+01;


    weight_hidden_output[0] = 5.056270956993103027e-01;
    weight_hidden_output[1] = -6.831505775451660156e+00;
    weight_hidden_output[2] = -8.631275892257690430e-01;
    weight_hidden_output[3] = -2.654483355581760406e-02;
    weight_hidden_output[4] = -2.393820555880665779e-03;
    weight_hidden_output[5] = 2.813248634338378906e-01;

    bias_output = 1.628709435462951660e+00;
}

#endif //__H_MODEL__
