# Sky Atmosphere Rendering Technique with Refraction  

![ReadMeComparison](https://github.com/ReasonPsycho/UnrealEngineSkyAtmosphere-Mirage/blob/master/ReadMeComperasion.png)  

This repository contains the codebase for my diploma dissertation, *"Image Synthesis of the Digital Environment Considering the Influence of Temperature on Light Ray Paths."*  

In this project, I attempted to simulate mirages by incorporating refraction based on temperature variations. However, using the SebH volumetric path tracer as a foundation turned out to be a poor choice for this approach. The long ray-marching steps and lack of true intersections with the ground prevented me from achieving a satisfying effect.  

## Modifications to the Original Project:  
- Implemented refraction based on temperature differences during null events.  
- Added collision detection based on ray-ground intersections.  
- Incorporated depth and diffuse maps based on real-world locations.  

If you're interested, you can read my dissertation in Polish [here](https://github.com/ReasonPsycho/UnrealEngineSkyAtmosphere-Mirage/blob/master/Image%20synthesis%20of%20the%20digital%20environment%20taking%20into%20account%20the%20influence%20of%20temperature%20on%20the%20light%20ray%20path%20by%20Krzysztof%20Czerwiński.pdf).  


# Orginal description - [Unreal Engine](https://www.unrealengine.com) Sky Atmosphere Rendering Technique

This is the project accompanying the paper `A Scalable and Production Ready Sky and Atmosphere Rendering Technique` presented at [EGSR 2020](https://egsr2020.london/program/).
It can be used to compare the new technique proposed in that paper to [Bruneton et al.](https://github.com/ebruneton/precomputed_atmospheric_scattering) as well as to a real-time path tracer.
The path tracer is using a simple and less efficient noise because I was not sure I could share noise code I had before.
The technique is used to render sky and atmosphere in [Unreal Engine](https://www.unrealengine.com).


Build the solution
1. Update git submodules (run `git submodule update --init`)
2. Open the solution 
3. Make sure you select a windows SDK and platform toolset you have locally for both projects
4. In Visual Studio, change the _Application_ project _Working Directory_ from `$(ProjectDir)` to `$(SolutionDir)`
5. Select Application as the startup project, hit F5

Runtime keys:
- SHIFT + mouse to look around
- CTRL  + mouse to move the sun around
- C to capture a screenshot
- T toggle between ray-marching and path tracing (disables the multiple scattering approximation when switch to path tracing)
- F5/F9 to save/load a state

Submodules
* [imgui](https://github.com/ocornut/imgui) V1.62 supported
* [tinyexr](https://github.com/syoyo/tinyexr)

About the code:
* The code of Eric Bruneton has been copied with authorization from his [2017 implementation](https://ebruneton.github.io/precomputed_atmospheric_scattering/).
* The code in this repository is provided in hope that later work using it to advance the state of the art will also be shared for every one to use.
* The code has been built from using the [Dx11Base](https://github.com/sebh/Dx11Base) demo/test platform.
* RenderSkyRayMarching.hlsl contains the ray marcher building the different LUTs for the new technique.
* RenderSkyPathTracing.hlsl contains the basic volumetric path tracer. It is matching PBRT and Mitsuba output for [other participating media tests](https://twitter.com/SebHillaire/status/1076144032961757185). It could be improved by really following the Radiance transfert Equation path integral as a loop for each event: currently we only handle participating media and intersection with the planet as a special case.
* This code has been tested on Windows only with visual studio (no build script generation)

Thanks to [Epic Games](https://www.epicgames.com) for allowing the release of this source code.

The code is provided as is and no support will be provided. 
I am still interested in any feedback, and we can discuss if you [contact me via twitter](https://twitter.com/SebHillaire).

[Seb](https://sebh.github.io/)
