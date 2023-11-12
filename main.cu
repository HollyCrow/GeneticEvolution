#include <iostream>
#include <stdio.h>
#include <SDL2/SDL.h>
#include <thread>
#include <cmath>

#define pi 3.141592
#define e 2.71828
#define input_number 20
#define internal_number 5
#define output_number 2
#define view_number 8 // Number of "viewer" lines, or "feelers", lingo pending
#define view_angle 0.78539816339
#define max_prey 100
#define max_predators 50
#define initial_prey 50
#define initial_predators 25
#define map_size_x 500
#define map_size_y 500
#define prey_max_speed 2
#define prey_max_turn_speed 0.1
#define predators_max_speed 2
#define predators_max_turn_speed 0.1
#define prey_view_distance 100
#define predators_view_distance 100

using namespace std::literals;
using clock_type = std::chrono::high_resolution_clock;

struct Network{
public:
    float input_to_internal_weight[internal_number][input_number];
    float internal_bias[internal_number];

    float internal_to_output_weight[output_number][internal_number];
    float output_bias[output_number];

    int internals[internal_number];

    float inputs[input_number];
    float outputs[output_number];
};
struct Agent{
public:
    float pos[2];
    float direction;
    float energy;
    float age;
    bool dead;

    Network network;
};
struct Data{
    Agent prey[max_prey];
    Agent predators[max_predators];
    int number_of_prey;
    int number_of_predators;
    int tick;
};

const int screen_width = 1000; // Variables for libs etc. Not needed to change much.
const int screen_height = 1000;
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;
bool running = true;
bool paused = false;
Data data;
Data *cuda_data;



__device__ void agent_update_initial(Agent* agent, int index){
    agent[index].age ++;
    agent[index].network.inputs[0] = agent[index].pos[0]; // Agent is spatially aware. This is probably worth testing on and off.
    agent[index].network.inputs[1] = agent[index].pos[1];
    agent[index].network.inputs[2] = agent[index].age;    // Probably will end up being completely irrelevant, will be interesting if they start acting different depending on age though.
    agent[index].network.inputs[3] = agent[index].energy;
    for (int n = 4; n < input_number; n++){ //Clear input values
        agent[index].network.inputs[n] = -1;
    }
}
__device__ void agent_think(Agent* agent, int index){
    for (int n = 0; n < internal_number; n++){ // Input -> internals
        agent[index].network.internals[n] = agent[index].network.internal_bias[n];
        for (int i = 0; i < input_number; i++){
            agent[index].network.internals[n] += agent[index].network.inputs[i]*agent[index].network.input_to_internal_weight[n][i]; // This feels like it might be the wrong way round
        }
        agent[index].network.internals[n] = 1 - (2 / (1 + pow(e, agent[index].network.internals[n]))); //Sigmoid (?)
    }
    for (int n = 0; n < output_number; n++){ // internal -> outputs
        agent[index].network.outputs[n] = agent[index].network.output_bias[n];
        for (int i = 0; i < internal_number; i++){
            agent[index].network.outputs[n] += agent[index].network.internals[i]*agent[index].network.internal_to_output_weight[n][i]; // This feels like it might be the wrong way round
        }
        agent[index].network.outputs[n] = 1 - (2 / (1 + pow(e, agent[index].network.outputs[n])));
    }
}
__device__ void agent_update_final(Agent* agent, int index){
    agent[index].direction += (agent[index].network.outputs[1])*prey_max_turn_speed;
    float speed = (agent[index].network.outputs[0])*prey_max_speed;
    agent[index].pos[0] += sin(agent[index].direction)*speed;
    agent[index].pos[1] += cos(agent[index].direction)*speed;

//    if (agent[index].pos[0] > map_size_x){agent[index].pos[0] -= map_size_x;            // Move agent off the board back to the other side
//    }else if (agent[index].pos[0] < map_size_x){agent[index].pos[0] += map_size_x;}     // Causes the agents to flash for some bizarre reason?
//    if (agent[index].pos[1] > map_size_y){agent[index].pos[1] -= map_size_y;}         // Disabled for now as it's not that important.
//    else if (agent[index].pos[1] < map_size_y){agent[index].pos[1] += map_size_y;}
}


__global__ void prey_process(Data* data) { // Unsure weather to merge this and predator_process into one function.
    int index = (blockIdx.x); //TODO work for thread and block split
//    if (index >= data->number_of_prey){index -= data->number_of_prey;} // Allows prey and predators to work on the same function.
    agent_update_initial(data->prey, index);

    for (int n = 0; n < data->number_of_predators; n++) // Set inputs. Decided to not move this to its own function for now.
    {
        if (n == index) continue;
        float distance[3] = {
                data->predators[n].pos[0]-data->prey[index].pos[0],
                data->predators[n].pos[1]-data->prey[index].pos[1],
                distance[0]*distance[0] + distance[1]*distance[1] // I cannot believe that it lets me do this
        };
        if (distance[2] < prey_view_distance*prey_view_distance){
            float angle = atan2(distance[0], distance[1]) - data->prey[index].direction;
            int closest = (angle/view_angle)+4;
            if (data->prey[index].network.inputs[closest+4] < (25/sqrt(distance[2]))){
                data->prey[index].network.inputs[closest+4] = (25/sqrt(distance[2]));
//                data->prey[index].network.inputs[closest+12] = 2; // Prey; 0, food; 1, predator; 2.
//                printf("New prey!: %f, %f, %d\n", data->prey[index].network.inputs[closest+4], sqrt(distance[2]), (int) data->prey[index].network.inputs[closest+12]);
            }
        }
    }
    agent_think(data->prey, index);
    agent_update_final(data->prey, index);
}
__global__ void predator_process(Data* data){
    int index = (blockIdx.x); //TODO work for thread and block split
//    if (index >= data->number_of_prey){index -= data->number_of_prey;} // Allows prey and predators to work on the same function.
    agent_update_initial(data->predators, index);

    for (int n = 0; n < data->number_of_prey; n++) // Set inputs. Decided to not move this to its own function for now.
    {
        if (n == index) continue;
        float distance[3] = {
                data->predators[index].pos[0]-data->prey[n].pos[0],
                data->predators[index].pos[1]-data->prey[n].pos[1],
                distance[0]*distance[0] + distance[1]*distance[1] // I cannot believe that it lets me do this
        };
        if (distance[2] < predators_view_distance*predators_view_distance){
            float angle = atan2(distance[0], distance[1]) - data->predators[index].direction;
            int closest = (angle/view_angle)+4;
            if (data->predators[index].network.inputs[closest+4] < (25/sqrt(distance[2]))){
                data->predators[index].network.inputs[closest+4] = (25/sqrt(distance[2]));
//                data->prey[index].network.inputs[closest+12] = 2; // Prey; 0, food; 1, predator; 2.
//                printf("New prey!: %f, %f, %d\n", data->prey[index].network.inputs[closest+4], sqrt(distance[2]), (int) data->prey[index].network.inputs[closest+12]);
            }
        }
    }
    agent_think(data->predators, index);
    agent_update_final(data->predators, index);
}


void FixedUpdate()// Fixed time updater
{
    auto target_time = clock_type::now() + 30ms;
    while (running) {
        if (!paused){
            //TODO actual processing
            prey_process<<<data.number_of_prey, 1>>>(cuda_data);
            predator_process<<<data.number_of_predators, 1>>>(cuda_data);
            cudaDeviceSynchronize();
            cudaMemcpy(&data, cuda_data, sizeof (data), cudaMemcpyDeviceToHost);
//            paused = true;
        }
        std::this_thread::sleep_until(target_time);
        target_time += 30ms;
    }
}
void draw() // Unlocked screen updater
{
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
void randomise_start() // Function to initially randomise the neural network and scatter the agents around the map.
{
    for (int n = 0; n < data.number_of_prey; n++){ //Scatter predators and prey
        data.prey[n].pos[0] = map_size_x*((float) rand())/((float)RAND_MAX);
        data.prey[n].pos[1] = map_size_y*((float) rand())/((float)RAND_MAX);

        for (int m = 0; m < internal_number; m++){ // Input -> internals
            data.prey[n].network.internal_bias[m] = (rand()/((float)RAND_MAX)) - 0.5;
            for (int i = 0; i < input_number; i++){
                data.prey[n].network.input_to_internal_weight[m][i] = (rand()/((float)RAND_MAX)) - 0.5;
            }
        }
        for (int m = 0; m < output_number; m++){ // internal -> outputs
            data.prey[n].network.output_bias[m] = (rand()/((float)RAND_MAX)) - 0.5;
            for (int i = 0; i < internal_number; i++){
                data.prey[n].network.internal_to_output_weight[m][i] = (rand()/((float)RAND_MAX)) - 0.5;
            }

        }
    }
    for (int n = 0; n < data.number_of_predators; n++){
        printf("Predator randomised: %d\n", n);
        data.predators[n].pos[0] = map_size_x*((float) rand())/((float)RAND_MAX);
        data.predators[n].pos[1] = map_size_y*((float) rand())/((float)RAND_MAX);

        for (int m = 0; m < internal_number; m++){ // Input -> internals
            data.predators[n].network.internal_bias[m] = rand()/((float)RAND_MAX) - 0.5;
            for (int i = 0; i < input_number; i++){
                data.predators[n].network.input_to_internal_weight[m][i] = (rand()/((float)RAND_MAX)) - 0.5;
            }
        }
        for (int m = 0; m < output_number; m++){ // internal -> outputs
            data.predators[n].network.output_bias[m] = rand()/((float)RAND_MAX) - 0.5;
            for (int i = 0; i < internal_number; i++){
                data.predators[n].network.internal_to_output_weight[m][i] = (rand()/((float)RAND_MAX)) - 0.5;
            }
        }
    }
}


int main(int argc, char **argv) {
    srand((unsigned) time(NULL));
    std::cout << "Genetic evolution!" << std::endl;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(screen_width, screen_height, 0, &window, &renderer);
    SDL_RenderSetScale(renderer,2,2);
    SDL_SetWindowTitle(window, "Genetic evolution");

    data.number_of_predators = initial_predators;
    data.number_of_prey = initial_prey;

    randomise_start();
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






