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
};


const int screen_width = 1000;
const int screen_height = 1000;
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

bool running = true;
bool paused = false;

void FixedUpdate(){ // Fixed time updater
    auto target_time = clock_type::now() + 30ms;
    while (running) {
        if (!paused){
//            printf("Fixed updating!\n");
        }
        std::this_thread::sleep_until(target_time);
        target_time += 30ms;
    }
}


void draw(){
    SDL_SetRenderDrawColor(renderer, 0,0,0,255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
}



int main(int argc, char **argv) {
    std::cout << "Genetic evolution!" << std::endl;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(screen_width, screen_height, 0, &window, &renderer);
    SDL_RenderSetScale(renderer,1,1);
    SDL_SetWindowTitle(window, "Genetic evolution");

    std::thread physicsThread(&FixedUpdate);


    std::vector<Agent> prey;
    std::vector<Agent> predators;


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


    return 0;
}
