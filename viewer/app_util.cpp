#include <iostream>
#include <string>
#include <SDL.h>
#include "util.h"

std::string get_base_path()
{
    std::string path;
    if (path.empty()) {
        char *base_path = SDL_GetBasePath();
        if (base_path) {
            path = base_path;
            SDL_free(base_path);
        } else {
            std::cout << "Error getting resource path: " << SDL_GetError() << std::endl;
            path = "./";
        }
    }
    return path;
}
