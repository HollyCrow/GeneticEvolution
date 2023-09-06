#include <iostream>
#include <stdio.h>
#include <SDL2/SDL.h>
#include <thread>
#include <vector>

#define max_prey 100
#define max_predators 50
#define initial_prey 50
#define initial_predators 25
#define map_size_x 500
#define map_size_y 500
#define prey_max_speed 1
#define prey_max_turn_speed 1
#define predators_max_speed 1
#define predators_max_turn_speed 1
#define prey_view_distance 30
#define predators_view_distance 50

#define input_number 20
#define internal_number 5
#define output_number 2


using namespace std::literals;
using clock_type = std::chrono::high_resolution_clock;


class Network{
private:
    float input_to_internal_weight[internal_number][input_number];
    float internal_bias[internal_number];

    float internal_to_output_weight[output_number][internal_number];
    float output_bias[output_number];

    int internals[internal_number];
public:
    int inputs[input_number];
    int outputs[output_number];


    void think(){
        for (int n = 0; n < internal_number; n++){ // Input -> internals
            internals[n] = internal_bias[n];
            for (int i = 0; i < input_number; i++){
                internals[n] += inputs[i]*input_to_internal_weight[n][i]; // This feels like it might be the wrong way round
            }
            internals[n] = internals[n] / (1 + abs(internals[n]));
        }
        for (int n = 0; n < output_number; n++){ // internal -> outputs
            outputs[n] = output_bias[n];
            for (int i = 0; i < internal_number; i++){
                outputs[n] += internals[i]*internal_to_output_weight[n][i]; // This feels like it might be the wrong way round
            }
            outputs[n] = outputs[n] / (1 + abs(outputs[n]));
        }
    }
};


class Agent{
public:
    float pos[2];
    float direction;
    float energy;
    float age;

    Network network;

    Agent(float x, float y){
        this->pos[0] = x;
        this->pos[1] = y;
    }
    Agent(){
        this->pos[0] = 100;
        this->pos[1] = 100;
    }
};


const int screen_width = 1000;
const int screen_height = 1000;
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

bool running = true;
bool paused = false;


struct Data{
    Agent prey[max_prey];
    Agent predators[max_predators];
    int number_of_prey;
    int number_predators;
    int tick;
};
Data data;
Data *cuda_data;


void draw(){
    SDL_SetRenderDrawColor(renderer, 0,0,0,255);
    SDL_RenderClear(renderer);

    SDL_SetRenderDrawColor(renderer, 0,255,0,255);
    for (int n = 0; n < data.number_of_prey; n++){
        SDL_RenderDrawPoint(renderer, (int) data.prey[n].pos[0], (int) data.prey[n].pos[1]);
    }
    SDL_SetRenderDrawColor(renderer, 255,0,0,255);
    for (int n = 0; n < data.number_of_prey; n++){
        SDL_RenderDrawPoint(renderer, (int) data.predators[n].pos[0], (int) data.predators[n].pos[1]);
    }

    SDL_RenderPresent(renderer);
}






__global__ void prey_process(Data* data) {
    int index = (blockIdx.x); //TODO work for thread and block split


    return;
}
__global__ void predator_process(Data* data) {


    return;
}

void FixedUpdate(){ // Fixed time updater
    auto target_time = clock_type::now() + 30ms;
    while (running) {
        if (!paused){
            //TODO actual processing
            prey_process<<<32, 1>>>(cuda_data);
            cudaDeviceSynchronize();
            cudaMemcpy(&data, cuda_data, sizeof (data), cudaMemcpyDeviceToHost);
        }
        std::this_thread::sleep_until(target_time);
        target_time += 30ms;
    }
}

int main(int argc, char **argv) {
    srand((unsigned) time(NULL));
    std::cout << "Genetic evolution!" << std::endl;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(screen_width, screen_height, 0, &window, &renderer);
    SDL_RenderSetScale(renderer,1,1);
    SDL_SetWindowTitle(window, "Genetic evolution");

    data.number_predators = initial_predators;
    data.number_of_prey = initial_prey;

    for (int n = 0; n < data.number_of_prey; n++){ //Scatter predators and prey
        data.prey[n].pos[0] = 1000*((float) rand())/(RAND_MAX);
        data.prey[n].pos[1] = 1000*((float) rand())/(RAND_MAX);
    }
    for (int n = 0; n < data.number_predators; n++){
        data.predators[n].pos[0] = 1000*((float) rand())/(RAND_MAX);
        data.predators[n].pos[1] = 1000*((float) rand())/(RAND_MAX);
    }


    cudaMalloc(&cuda_data, sizeof (data));
    cudaMemcpy(cuda_data, &data, sizeof (data), cudaMemcpyHostToDevice);


    std::thread physicsThread(&FixedUpdate);

    while (running) {
        draw();
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym ==SDLK_SPACE)paused = !paused;
                    break;
            }
        }
    }

    physicsThread.join();



    return 0;
}
