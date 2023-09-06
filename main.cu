#include <iostream>
#include <stdio.h>
#include <SDL2/SDL.h>
#include <thread>

#define max_prey 100
#define max_predators 50
#define map_size_x 500
#define map_size_y 500
#define prey_max_speed 1
#define prey_max_turn_speed 1
#define predators_max_speed 1
#define predators_max_turn_speed 1

using namespace std::literals;
using clock_type = std::chrono::high_resolution_clock;

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
//    printf("Draw to screen\n");
}



int main(int argc, char **argv) {
    std::cout << "Genetic evolution!" << std::endl;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(screen_width, screen_height, 0, &window, &renderer);
    SDL_RenderSetScale(renderer,1,1);
    SDL_SetWindowTitle(window, "Genetic evolution");

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
                    if (event.key.keysym.sym ==SDLK_SPACE){
                        paused = !paused;
                    }
                    break;
            }
        }
    }


    return 0;
}
