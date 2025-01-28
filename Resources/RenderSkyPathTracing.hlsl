// Copyright Epic Games, Inc. All Rights Reserved.

#include "./Resources/RenderSkyCommon.hlsl"

////////////////////////////////////////////////////////////
// Sampling terrain
////////////////////////////////////////////////////////////

Texture2D<float4> TerrainHeightmapTex : register(t0);
Texture2D<float4> TerrainNormalmapTex : register(t9);
Texture2D<float4> TemperatureMap : register(t10);


bool IsBelowTerrain(in float3 vectorPos, out float3 terrainNormal)
{
    float3 actualPos = float3(vectorPos.x, vectorPos.y, vectorPos.z);
    const float maxTerrainHeight = 6359.0f;
    const float globalTerrainWidth = 5.0f;
    const float textureTerrainWidth = 1.0f;
    const float offsetX = -2.0f;
    const float offsetY = -2.0f;
    if (abs(actualPos.x + offsetX) >= globalTerrainWidth || abs(actualPos.y + offsetY) >= globalTerrainWidth)
        return false;

    // Normalize world position to UV space
    float2 normalizedPos = float2(
        ((actualPos.x + offsetX) + globalTerrainWidth) / (2 * globalTerrainWidth),
        ((actualPos.y + offsetY) + globalTerrainWidth) / (2 * globalTerrainWidth));

    float2 localUvs = float2(
        normalizedPos.x * textureTerrainWidth,
        normalizedPos.y * textureTerrainWidth
    );

    // Sample the height at the terrain location
    const float height = TerrainHeightmapTex.SampleLevel(samplerLinearClamp, localUvs, 0).r * 4 + maxTerrainHeight;

    // Compute terrain normal using the helper function
    terrainNormal = TerrainNormalmapTex.SampleLevel(samplerLinearClamp, localUvs, 0).rgb;

    return length(height - length(actualPos)) < 1;
}

float light_radiance(float wavelength)
{
    // Blackbody radiator at 5800 K (simplified exponential-like behavior).
    if (wavelength < 500.0f) return 1.0f; // Blue light
    if (wavelength < 600.0f) return 0.9f; // Green light
    return 0.8f; // Red light
}

////////////////////////////////////////////////////////////
// Path tracing context used by the integrators
////////////////////////////////////////////////////////////


struct PathTracingContext
{
    AtmosphereParameters Atmosphere;

    Ray ray;
    float3 P;
    float3 V; // not always the view: sometimes it is the opposite of ray.d when one bounce has happened.
    float3 N; // normal of the opaque ground

    float lastDiffrence;

    float scatteringMajorant;
    float extinctionMajorant;

    float3 wavelengthMask; // light source mask

    uint2 screenPixelPos;
    float randomState;

    int lastSurfaceIntersection;

    bool debugEnabled;
    bool singleScatteringRay;
    bool opaqueHit;

    bool hasScattered;
    float transmittance;
};

float random01(inout PathTracingContext ptc)
{
    // Trying to do the best noise here with simple function.
    // See https://www.shadertoy.com/view/ldjczd.
    float rnd = whangHashNoise(ptc.randomState, ptc.screenPixelPos.x, ptc.screenPixelPos.y);

    // This is some bad noise because not low discrepancy. I use such a noise for the EGSR comparisons but then it seems I forgot to checkin the code.

    //ptc.randomState++; return rnd;

#if 1
    //ptc.randomState += gTimeSec + gFrameId;	// hard to converge
    //ptc.randomState += gFrameId;
    ptc.randomState += gFrameId * 1280; // less pattern a the beginning be goes super wrong after a while...
#else
	uint animation = uint(gTime*123456.0);
	ptc.randomState += float((animation * 12345u) % 256u);
#endif

    return rnd;
}


////////////////////////////////////////////////////////////
// Misc functions
////////////////////////////////////////////////////////////


#define D_INTERSECTION_MEDIUM	0		// Participating media is going to be considered.
#define D_INTERSECTION_NULL		1		// Intersection with a null material surface surch as the top of the atmosphere.
#define D_INTERSECTION_GROUND	2		// Could be SOLID or SURFACE but let's call it ground in this particular case.

bool getNearestIntersection(inout PathTracingContext ptc, in Ray ray, inout float3 P)
{
    float3 earthO = float3(0.0f, 0.0f, 0.0f);
    float t = raySphereIntersectNearest(ray.o, ray.d, earthO, ptc.Atmosphere.BottomRadius);
    float tTop = raySphereIntersectNearest(ray.o, ray.d, earthO, ptc.Atmosphere.TopRadius);

    ptc.lastSurfaceIntersection = D_INTERSECTION_MEDIUM;
    if (t < 0.0f)
    {
        // No intersection with earth: use a super large distance

        if (tTop < 0.0f)
        {
            t = 0.0f; // No intersection with earth nor atmosphere: stop right away 
            return false;
        }
        else
        {
            ptc.lastSurfaceIntersection = D_INTERSECTION_NULL;
            t = tTop;
        }
    }
    else
    {
        if (tTop > 0.0f)
        {
            t = min(tTop, t);
        }
        ptc.lastSurfaceIntersection = t == tTop ? D_INTERSECTION_NULL : D_INTERSECTION_GROUND;
    }

    P = ray.o + t * ray.d;
    return true; // mark as always valid
}


////////////////////////////////////////////////////////////
// Volume functions
////////////////////////////////////////////////////////////


bool insideAnyVolume(in PathTracingContext ptc, in Ray ray)
{
    const float h = length(ray.o);
    if ((h - ptc.Atmosphere.BottomRadius) < PLANET_RADIUS_OFFSET)
        return false;
    if ((ptc.Atmosphere.TopRadius - h) < -PLANET_RADIUS_OFFSET)
        return false;
    return true;
}

struct MediumSample
{
    float scattering;
    float absorption;
    float extinction;

    float scatteringMie;
    float absorptionMie;
    float extinctionMie;

    float scatteringRay;
    float absorptionRay;
    float extinctionRay;

    float scatteringOzo;
    float absorptionOzo;
    float extinctionOzo;

    float albedo;
};

MediumSample sampleMedium(in PathTracingContext ptc)
{
    const float3 WorldPos = ptc.P;
    MediumSampleRGB medium = sampleMediumRGB(WorldPos, ptc.Atmosphere);

    MediumSample s;
    s.scatteringMie = dot(ptc.wavelengthMask, medium.scatteringMie);
    s.absorptionMie = dot(ptc.wavelengthMask, medium.absorptionMie);
    s.extinctionMie = dot(ptc.wavelengthMask, medium.extinctionMie);

    s.scatteringRay = dot(ptc.wavelengthMask, medium.scatteringRay);
    s.absorptionRay = dot(ptc.wavelengthMask, medium.absorptionRay);
    s.extinctionRay = dot(ptc.wavelengthMask, medium.extinctionRay);

    s.scatteringOzo = dot(ptc.wavelengthMask, medium.scatteringOzo);
    s.absorptionOzo = dot(ptc.wavelengthMask, medium.absorptionOzo);
    s.extinctionOzo = dot(ptc.wavelengthMask, medium.extinctionOzo);

    s.scattering = dot(ptc.wavelengthMask, medium.scattering);
    s.absorption = dot(ptc.wavelengthMask, medium.absorption);
    s.extinction = dot(ptc.wavelengthMask, medium.extinction);
    s.albedo = dot(ptc.wavelengthMask, medium.albedo);

    return s;
}


float basedLog(float base, float x)
{
    return log(x) / log(base);
}

float SampleTemperature( 
    float3 p,                // Point in space [x, y, z].
    Texture2D texture_1d,    // 2D texture containing the red channel values (normalized 0–1).
    float precision,         // Altitude resolution per pixel (in kilometers). 0.0001
    float height,      // Number of pixels in the 1D texture (provided from CPU). 5 000 000
    float minTemperatureF,   // Minimum temperature mapped to red value 0. -100.0
    float maxTemperatureF    // Maximum temperature mapped to red value 255. 100.0
)
{

    float seeLevel = 6361.0002f;
    float diffrence = 3.0f;

    /*http://fisicaatmo.at.fcen.uba.ar/practicas/ISAweb.pdf*/

    float t = saturate((p.z - seeLevel) / diffrence);
    //return 288.15+(-0.0065) *( pos.z + seeLevel) * 1000;
    return lerp(12.0f, 20.0f, t) + 274.15f;

    
    const float seaLevel = 6360.0f;     // Compute the total altitude range in kilometers covered by the texture
    const float max_texture_width = 16384;

    float pixels  = floor(height / precision); // Number of pixels in the 2D texture;
    const int max_texture_height = floor(pixels / max_texture_width) * 100;

    float3 position = p;
    position.z -= seaLevel;
    // Map z-coordinate (height in meters) to texture position in [0, 1]
    float pixel = floor(length(position) / precision); // also just height
    
    float2 textureCoord = float2(
        (pixel % max_texture_width) / max_texture_width,
       floor(pixel / max_texture_width) / max_texture_height
    );
    textureCoord = clamp(textureCoord, float2(0.0, 0.0), float2(1.0, 1.0)); // Clamp only for safety

    // Sample the texture at the computed texture coordinate
    float redValue = texture_1d.SampleLevel(samplerLinearClamp, textureCoord,0).r; // Red channel (normalized to 0-1)

    // Reverse normalization: Convert redValue (0–1) back to temperature in Fahrenheit
    float temperatureF = lerp(minTemperatureF, maxTemperatureF, redValue);

    return temperatureF;
}

float TemperatureIOR(in float tmp1, in float tmp2)
{
    return 1.0 + 0.000292 * saturate(tmp1 / max(tmp2, 0.001)); // Avoid division by near-zero
}

float3 TemperatureNormal(float3 direction, float3 p1, float3 p2) {
    // Small step for finite difference
    float delta = 0.0001f;

    // Gradient in the x direction
    float t1x = TemperatureIOR(SampleTemperature(p1,TemperatureMap, 0.0001, 50, -100.0,100.0), SampleTemperature(p2,TemperatureMap, 0.0001, 50, -100.0,100.0));
    float t2x = TemperatureIOR(SampleTemperature(p1,TemperatureMap, 0.0001, 50, -100.0,100.0), SampleTemperature(p2 + float3(delta, 0.0, 0.0),TemperatureMap, 0.0001, 5000000, -100.0,100.0));
    float gradient_x = (t2x - t1x) / delta;

    // Gradient in the y direction
    float t1y = TemperatureIOR(SampleTemperature(p1,TemperatureMap, 0.0001, 50, -100.0,100.0), SampleTemperature(p2,TemperatureMap, 0.0001, 50, -100.0,100.0));
    float t2y = TemperatureIOR(SampleTemperature(p1,TemperatureMap, 0.0001, 50, -100.0,100.0), SampleTemperature(p2 + float3(0.0, delta, 0.0),TemperatureMap, 0.0001, 5000000, -100.0,100.0));
    float gradient_y = (t2y - t1y) / delta;

    // Gradient in the z direction
    float t1z = TemperatureIOR(SampleTemperature(p1,TemperatureMap, 0.0001, 50, -100.0,100.0), SampleTemperature(p2,TemperatureMap, 0.0001, 50, -100.0,100.0));
    float t2z = TemperatureIOR(SampleTemperature(p1,TemperatureMap, 0.0001, 50, -100.0,100.0), SampleTemperature(p2 + float3(0.0, 0.0, delta),TemperatureMap, 0.0001, 5000000, -100.0,100.0));
    float gradient_z = (t2z - t1z) / delta;

    // Small value to avoid division by zero
    float epsilon = 1e-8;

    // Combine gradients and normalize
    float3 gradient = float3(
        -direction.x + gradient_x,
        -direction.y + gradient_y,
        -direction.z + gradient_z
    );

    float gradient_norm = length(gradient) + epsilon;

    return gradient / gradient_norm;
}

////////////////////////////////////////////////////////////
// Transmittance integrator
////////////////////////////////////////////////////////////


// Forward declaration
float Transmittance(
    inout PathTracingContext ptc,
    in float3 P0,
    in float3 P1,
    in float3 direction);

// Estimate a transmittance from ptc.P towards a direction.
float TransmittanceEstimation(in PathTracingContext ptc, in float3 direction)
{
    float beamTransmittance = 1.0f;
    float3 P0 = ptc.P + direction * RAYDPOS;
    float3 P1;

    float3 earthO = float3(0.0, 0.0, 0.0);
    float t = raySphereIntersectNearest(P0, direction, earthO, ptc.Atmosphere.BottomRadius);
    if (t > 0.0f)
    {
        beamTransmittance = 0.0f; // earth is intersecting
    }
    else
    {
        t = raySphereIntersectNearest(P0, direction, earthO, ptc.Atmosphere.TopRadius);
        P1 = P0 + t * direction;
        beamTransmittance = Transmittance(ptc, P0, P1, direction);
    }

    return beamTransmittance;
}


////////////////////////////////////////////////////////////
// Sampling functions
////////////////////////////////////////////////////////////


#define D_SCATT_TYPE_NONE 0
#define D_SCATT_TYPE_MIE  1
#define D_SCATT_TYPE_RAY  2
#define D_SCATT_TYPE_UNI  3
#define D_SCATT_TYPE_ABS  4	// Not really a scattering event: the trace has ended because absorbed

float calcHgPhasepdf(in PathTracingContext ptc, float cosTheta)
{
    return hgPhase(ptc.Atmosphere.MiePhaseG, cosTheta);
}

float calcHgPhaseInvertcdf(in PathTracingContext ptc, float zeta)
{
    const float G = ptc.Atmosphere.MiePhaseG;
    float one_plus_g2 = 1.0f + G * G;
    float one_minus_g2 = 1.0f - G * G;
    float one_over_2g = 0.5f / G;
    float t = (one_minus_g2) / (1.0f - G + 2.0f * G * zeta);
    return one_over_2g * (one_plus_g2 - t * t); // Careful: one_over_2g undefined for g~=0
}

void phaseEvaluateSample(in PathTracingContext ptc, in float3 sampleDirection, in int ScatteringType, out float value,
                         out float pdf)
{
    const float3 wi = sampleDirection;
    const float3 wo = ptc.V;
    float cosTheta = dot(wi, wo);

    if (ScatteringType == D_SCATT_TYPE_RAY)
    {
        value = RayleighPhase(cosTheta);
        pdf = 0.25 / PI; // 1/4PI since no importance is used in this case.
    }
    else //if (ScatteringType == D_SCATT_TYPE_MIE)
    {
        value = hgPhase(ptc.Atmosphere.MiePhaseG, cosTheta);
#if MIE_PHASE_IMPORTANCE_SAMPLING
		pdf = value;
#else
        pdf = 0.25 / PI; // 1/4PI since phaseGenerateSample is not importance sampling a direction.
#endif
    }
    //else // D_SCATT_TYPE_UNI
    //{
    //	pdf = uniformPhase();
    //	value = pdf;
    //}
}

void phaseGenerateSample(inout PathTracingContext ptc, out float3 newDirection, in int ScatteringType, out float value,
                         out float pdf)
{
    if (ScatteringType == D_SCATT_TYPE_RAY)
    {
        // Evaluate a random direction
        newDirection = getUniformSphereSample(random01(ptc), random01(ptc));
    }
    else //if (ScatteringType == D_SCATT_TYPE_MIE)
    {
#if MIE_PHASE_IMPORTANCE_SAMPLING
		// Evaluate a random direction with importance sampling
		float phi = 2.0f * PI * random01(ptc);
		float cosTheta = calcHgPhaseInvertcdf(ptc, random01(ptc));
		float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));	// max to make sqrt safe. Can make the GPU hang otherwise...

		const float3 wo = ptc.V;
		float3 mainDir = -wo;		// -wo, we use the same invertcdf convention as in http://www.pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Sampling_Volume_Scattering.html#fragment-ComputedirectionmonowiforHenyey--Greensteinsample-0
		float3 t0, t1;

		const float3 up = float3(0.0, 1.0, 0.0);
		const float3 right = float3(1.0, 0.0, 0.0);
		if (abs(dot(mainDir, up)) > 0.01)
		{
			t0 = normalize(cross(mainDir, up));
			t1 = normalize(cross(mainDir, t0));
		}
		else
		{
			t0 = normalize(cross(mainDir, right));
			t1 = normalize(cross(mainDir, t0));
		}

		newDirection = sinTheta * sin(phi) * t0 + sinTheta * cos(phi) * t1 + cosTheta * mainDir;
		//newDirection = normalize(newDirection);
#else
        newDirection = getUniformSphereSample(random01(ptc), random01(ptc));
#endif
    }
    //else // D_SCATT_TYPE_UNI
    //{
    //	// Evaluate a random direction
    //	newDirection = getUniformSphereSample(random01(ptc), random01(ptc));
    //}

    // From direction, evaluate the phase value and pdf
    phaseEvaluateSample(ptc, newDirection, ScatteringType, value, pdf);
}


void lightGenerateSample(inout PathTracingContext ptc, out float3 direction, out float value, out float pdf,
                         out float beamTransmittance, out bool isDeltaLight)
{
    direction = sun_direction;
    pdf = 1;
    isDeltaLight = true;

    value = dot(gSunIlluminance, ptc.wavelengthMask); // Value not taking into account trasmittance

    beamTransmittance = TransmittanceEstimation(ptc, direction); // Compute transmitance through atmosphere
}


////////////////////////////////////////////////////////////
// Integrators
////////////////////////////////////////////////////////////


#if TRANSMITANCE_METHOD == 0

//// Delta tracking
float Transmittance(
	inout PathTracingContext ptc,
	in float3 P0,
	in float3 P1,
	in float3 direction)
{
#if SHADOWMAP_ENABLED
	// First evaluate opaque shadow
	if (getShadow(ptc.Atmosphere, P0) <= 0.0)
		return 0.0;
#endif
	float distance = length(P0 - P1);

	bool terminated = false;
	float t = 0;
	do
	{
		float zeta = random01(ptc);
		t = t + infiniteTransmittanceIS(ptc.extinctionMajorant, zeta);

		if (t > distance)
			break; // Did not terminate in the volume

		// Update the shading context
		float3 P = P0 + t * direction;
		ptc.P = P;

		// Evaluate the local absorption after updating the shading context
		float extinction = sampleMedium(ptc).extinction;
		float xi = random01(ptc);
		if (xi < (extinction / ptc.extinctionMajorant))
		{
			terminated = true;
		}

#if DEBUGENABLED // Transmittance estimation point
		if (ptc.debugEnabled)
		{
			float3 color = terminated ? float3(0.5, 0.0, 0.0) : float3(0.0, 0.5, 0.0);
			addGpuDebugCross(ToDebugWorld + ptc.P, color, 1.0);
			addGpuDebugLine(ToDebugWorld + P0, ToDebugWorld + ptc.P, float3(0.5, 0.5, 0.5));
		}
#endif
	} while (!terminated);

	if (terminated)
		return 0.0;
	else
		return 1.0;
}

#elif TRANSMITANCE_METHOD == 1

//// Ratio tracking from http://drz.disneyresearch.com/~jnovak/publications/RRTracking/index.html
float Transmittance(
	inout PathTracingContext ptc,
	in float3 P0,
	in float3 P1,
	in float3 direction)
{
#if SHADOWMAP_ENABLED
	// First evaluate opaque shadow
	if (getShadow(ptc.Atmosphere, P0) <= 0.0)
		return 0.0;
#endif
	float distance = length(P0 - P1);
	float3 dir = float3(P1 - P0) / distance;

	float t = 0;
	float transmittance = 1.0f;
	do
	{
		float zeta = random01(ptc);
		t = t + infiniteTransmittanceIS(ptc.extinctionMajorant, zeta);

		// Update the shading context
		float3 P = P0 + t * dir;
		ptc.P = P;

		if (t > distance)
			break; // Did not terminate in the volume

		float extinction = sampleMedium(ptc).extinction;
		transmittance *= 1.0f - max(0.0f, extinction / ptc.extinctionMajorant);

#if DEBUGENABLED // Transmittance estimation point
		if (ptc.debugEnabled)
		{
			float3 color = lerp(float3(0.0, 0.5, 0.0), float3(0.5, 0.0, 0.0), float3(transmittance, transmittance, transmittance));
			addGpuDebugCross(ToDebugWorld + ptc.P, color, 1.0);
			addGpuDebugLine(ToDebugWorld + P0, ToDebugWorld + ptc.P, float3(0.5, 0.5, 0.5));
		}
#endif
	} while (true);
	return saturate(transmittance);
}

#elif TRANSMITANCE_METHOD == 2

float Transmittance(
    inout PathTracingContext ptc,
    in float3 P0,
    in float3 P1,
    in float3 direction)
{
#if SHADOWMAP_ENABLED
	// First evaluate opaque shadow
	if (getShadow(ptc.Atmosphere, P0) <= 0.0)
		return 0.0;
#endif

    // Second evaluate transmittance due to participating media
    float viewHeight = length(P0);
    const float3 UpVector = P0 / viewHeight;
    float viewZenithCosAngle = dot(direction, UpVector);
    float2 uv;
    LutTransmittanceParamsToUv(ptc.Atmosphere, viewHeight, viewZenithCosAngle, uv);
    const float3 trans = TransmittanceLutTexture.SampleLevel(samplerLinearClamp, uv, 0).rgb;

#if DEBUGENABLED // Transmittance value
	if (ptc.debugEnabled)
	{
		addGpuDebugLine(ToDebugWorld + P0, ToDebugWorld + P0 + direction*100.0f, trans);
	}
#endif

    return dot(trans, ptc.wavelengthMask);
}

#else

#error Transmittance needs to be implemented.

#endif


bool Integrate(
    inout PathTracingContext ptc,
    in Ray wi,
    inout float3 P, // closestHit
    out float L,
    out float transmittance,
    out float weight,
    inout Ray wo,
    inout int OutScatteringType)
{
    Ray localWi = wi;

    OutScatteringType = D_SCATT_TYPE_NONE;

    float3 P0 = ptc.P;
    if (!getNearestIntersection(ptc, createRay(P0, localWi.d), P))
        return false;
    float tMax = length(P - P0);

    /*
    if (ptc.singleScatteringRay) //Orginal collision with the ground
    {		
        float2 pixPos = ptc.screenPixelPos; 
        float3 ClipSpace = float3((pixPos / float2(gResolution))*float2(2.0, -2.0) - float2(1.0, -1.0), 0.5);
        ClipSpace.z = ViewDepthTexture[pixPos].r;
        if (ClipSpace.z < 1.0f)
        {
            ptc.opaqueHit = true;
            float4 DepthBufferWorldPos = mul(gSkyInvViewProjMat, float4(ClipSpace, 1.0));
            DepthBufferWorldPos /= DepthBufferWorldPos.w;

            float tDepth = length(DepthBufferWorldPos.xyz - (ptc.P+float3(0.0, 0.0, -ptc.Atmosphere.BottomRadius))); // apply earth offset to go back to origin as top of earth mode.
            if (tDepth < tMax)
            {
                // P and ptc.P will br written in so no need to update them
                tMax = tDepth;
            }
        }
    }
    */


    bool eventScatter = false;
    bool eventAbsorb = false;
    float extinction = 0;
    float scattering = 0;
    float albedo = 0;
    transmittance = 1.0f;

    float t = 0;
    do
    {
        if (ptc.extinctionMajorant == 0.0) break; // cannot importance sample, so stop right away

        if (t >= tMax || ptc.opaqueHit)
        {
            break; // Did not terminate in the volume
        }
        
        float zeta = random01(ptc);
        t = t + infiniteTransmittanceIS(ptc.extinctionMajorant, zeta);

        // Update the shading context
        float3 P1 = P0 + t * localWi.d;
        ptc.P = P1;

        if (ptc.singleScatteringRay)
        {
            ptc.opaqueHit = IsBelowTerrain(ptc.P, ptc.N);
        }

        float lastDiffrence = 0;
#if DEBUGENABLED // Sample point
		if (ptc.debugEnabled) { addGpuDebugCross(ToDebugWorld + ptc.P, float3(0.5, 1.0, 1.0), 0.5); }
#endif

        // Recompute the local extinction after updating the shading context
        MediumSample medium = sampleMedium(ptc);
        extinction = medium.extinction;
        scattering = medium.scattering;
        albedo = medium.albedo;
        float xi = random01(ptc);
        if (xi <= (medium.scattering / ptc.extinctionMajorant))
        {
            eventScatter = true;

            float zeta = random01(ptc);
            if (zeta < medium.scatteringMie / medium.scattering)
            {
                OutScatteringType = D_SCATT_TYPE_MIE;
            }
            else
            {
                OutScatteringType = D_SCATT_TYPE_RAY;
            }
        }
        else if (xi < (medium.extinction / ptc.extinctionMajorant))
        // on top of scattering, as extinction = scattering + absorption
            eventAbsorb = true;

        else // null event
        {
            float t1 = SampleTemperature(P0,TemperatureMap, 0.0001, 50, -100.0,100.0);
            float t2 = SampleTemperature(P1,TemperatureMap, 0.0001, 50, -100.0,100.0);
            float currentDifference = abs(t2 - t1) + lastDiffrence;

            if (currentDifference > 0.001)
            {
                P1 = P0 + localWi.d * 0.001;
            
                t1 = SampleTemperature(P0,TemperatureMap, 0.0001, 50, -100.0,100.0);
                t2 = SampleTemperature(P1,TemperatureMap, 0.0001, 50, -100.0,100.0);
                do
                {
                    float eta = TemperatureIOR(t1, t2);
                    float3 normal = TemperatureNormal(localWi.d, P0, P1);
                    localWi.d = refract(localWi.d, normal, eta);
                    P0 = P1;
                    P1 = P0 + localWi.d * 0.001;
                    ptc.P = P1;
                    lastDiffrence = 0;
                    t1 = SampleTemperature(P0,TemperatureMap, 0.0001, 50, -100.0,100.0);
                    t2 = SampleTemperature(P1,TemperatureMap, 0.0001, 50, -100.0,100.0);
                    currentDifference = abs(t2 - t1);
                }while (currentDifference > 0.001);
                
            }else
            {
                lastDiffrence = currentDifference;
            }    
            //addGpuDebugLine(ptc.P, ptc.P + refractedRay * 10.0f, float3(1.0f, 0.0f, 0.0f));
        }
    }
    while (!(eventScatter || eventAbsorb));

    if (eventScatter && all(extinction > 0.0))
    {
#if DEBUGENABLED // Path
		if (ptc.debugEnabled) { addGpuDebugLine(ToDebugWorld + P0, ToDebugWorld + ptc.P, float3(0, 1, 0)); }
#endif
        ptc.lastSurfaceIntersection = D_INTERSECTION_MEDIUM;
        P = ptc.P;

        const float Tr = 1.0;
        // transmittance to previous event (view since in this case we only consider single scattering)
        const float pdf = infiniteTransmittancePDF(scattering, Tr);
        // This must use scattering since this is the scattering pdf case after scatter/absorb separation
        transmittance = Tr;

        // Note that pdf already has extinction in it, so we should avoid the multiply and divide; it is shown here for clarity
        weight = albedo * extinction / pdf;
    }
    else if (eventAbsorb)
    {
#if DEBUGENABLED // Path
		if (ptc.debugEnabled) { addGpuDebugLine(ToDebugWorld + P0, ToDebugWorld + ptc.P, float3(0, 1, 0)); }
#endif
        OutScatteringType = D_SCATT_TYPE_ABS;
        ptc.lastSurfaceIntersection = D_INTERSECTION_MEDIUM;
        P = ptc.P;

        transmittance = 0.0; // will set throughput to 0 and stop processing loop
        weight = 0.0; // will remove lighting
    }
    else
    {
        // Max distance reached without absorption or scattering event. Keep lastSurfaceIntersection computed in getNearestIntersection above.
        P = P0 + tMax * localWi.d; // out of the volume range

#if DEBUGENABLED // Path
		if (ptc.debugEnabled) { addGpuDebugLine(ToDebugWorld + P0, ToDebugWorld + P, float3(0, 1, 0)); }
#endif

        transmittance = 1.0f;
        float pdf = transmittance;
        weight = 1.0 / pdf;
    }

    L = 0.0f;
    wo = createRay(P + localWi.d * RAYDPOS, localWi.d);

    return true;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


// This is the volume path tracer core loop. It was tested on other volumetric data and compared against PBRT and Mitsuba (e.g. https://twitter.com/SebHillaire/status/1076144032961757185 or https://twitter.com/SebHillaire/status/1073568200762245122)
// It could be improved by really following the Radiance transfert Equation path integral as a loop for each event.
float2 LightIntegratorInner(
    VertexOutput Input,
    inout PathTracingContext ptc)
{
    //////////
    float2 pixPos = Input.position.xy;
    float3 ClipSpace = float3((pixPos / float2(gResolution)) * float2(2.0, -2.0) - float2(1.0, -1.0), 0.5);
    float4 HPos = mul(gSkyInvViewProjMat, float4(ClipSpace, 1.0));

    float earthR = ptc.Atmosphere.BottomRadius;
    float3 earthO = float3(0.0, 0.0, 0.0);

    float3 WorldDir = normalize(HPos.xyz / HPos.w - camera);
    float3 camPos = camera + float3(0, 0, earthR);
    float3 sunDir = sun_direction;


    float DiffuseReturnValue = 0.0f;
    //////////

    float L = 0.0f;
    float throughput = 1.0f;
    Ray ray = createRay(camPos, WorldDir); // ray from camera to pixel

    // Path tracer loop
    // Based on Production Volume Rendering https://graphics.pixar.com/library/ProductionVolumeRendering/

    //float t = raySphereIntersectNearest(camPos, WorldDir, float3(0.0f, 0.0f, 0.0f), ptc.Atmosphere.BottomRadius); if (t > 0.0f) {	return 0.0;	} // hit earth: stop tracing, in game can be a test against the depth buffer

    // Move to top atmospehre
    float3 prevRayO = ray.o;
    if (MoveToTopAtmosphere(ray.o, ray.d, ptc.Atmosphere.TopRadius))
    {
#if DEBUGENABLED // Atmosphere entrance
		if (ptc.debugEnabled) { addGpuDebugLine(ToDebugWorld + prevRayO, ToDebugWorld + ray.o, float3(0, 0, 1)); }
#endif
        camPos = ray.o;
    }
    else
    {
        // Ray is not intersecting the atmosphere
        return L;
    }

    float3 P = ray.o; // Default start point when the camera is inside a volume
    float3 prevDebugPos = P;

    bool hasScattered = false;
    int step = 0;
    while (step < gScatteringMaxPathDepth && throughput > 0.0)
    {
#if true //  TODO add again GROUND_GI_ENABLED and edge case opaque
        if ((ptc.lastSurfaceIntersection == D_INTERSECTION_GROUND || ptc.opaqueHit) && !hasScattered)
        // ptc.hasScattered is checked to avoid colored ground
        {
            // If ground is directly visible as the first intersection (has not scattered before) then we should stop tracing for the ground to not show up.
            break;
        }

        // Handle collision with opaque
        if ((ptc.lastSurfaceIntersection == D_INTERSECTION_GROUND || ptc.opaqueHit) && hasScattered)
        // ptc.hasScattered is checked to avoid colored ground
        {
            // Offset position to be always be the volume 
            float h = length(ray.o);
            float3 UpVector = ray.o / h;
            ray.o = ray.o + UpVector * 0.025f;
            P = ray.o;

            // Could also add emissive contribution to luminance here to simulate city lights.
            float SunTransmittance = TransmittanceEstimation(ptc, sunDir);

            // Generate a new up direction assuming a diffuse surface (incorrectly as it would need to follow a cosine distribution with matching pdf, but ok as it is not used for comparison images)
            float3 newDirection = getUniformSphereSample(random01(ptc), random01(ptc));
            const float dotVec = dot(newDirection, UpVector);
            if (dotVec < 0.0f)
            {
                newDirection = newDirection - 2.0f * dotVec * UpVector;
            }
            ray.d = newDirection;
            //const float NdotV = saturate(dotVec);
            const float NdotL = saturate(dot(UpVector, sunDir));
            const float albedo = saturate(dot(ptc.Atmosphere.GroundAlbedo, ptc.wavelengthMask));
            const float DiffuseEval = albedo * (1.0f / PI);
            const float DiffusePdf = 1;

            // Also update throughput based on albedo.
            L += throughput * SunTransmittance * (DiffuseEval * NdotL * dot(gSunIlluminance, ptc.wavelengthMask));
            throughput *= DiffuseEval / DiffusePdf;

#if 0 //DEBUGENABLED // Bounce on ground 
			if (ptc.debugEnabled)
			{
				float3 DebugP = ToDebugWorld + ray.o + UpVector * 0.1;
				addGpuDebugLine(DebugP, DebugP + newDirection * 10, float3(1, 1, 1));
				addGpuDebugCross(ToDebugWorld + ray.o, float3(0, 0, 1), 10.0);
			}
#endif
        }
#endif

        // store current context: ray, intersection point P, etc.
        ptc.ray = ray;
        ptc.P = P + ray.d * RAYDPOS;
        ptc.V = -ray.d;

        // Compute next ray from last intersection.
        // From there, next ray is the reference ray for volumetric interactions.
        Ray nextRay = createRay(P + ray.d * RAYDPOS, ray.d);
        int ScatteringType = D_SCATT_TYPE_NONE;

        if (insideAnyVolume(ptc, nextRay))
        {
            float Lv = 0.0;
            float transmittance = 0.0;
            float weight = 0.0;

            bool hasCollision = Integrate(ptc, nextRay, P, Lv, transmittance, weight, nextRay, ScatteringType);
            // Run volume integrator on the considered range

#if  DEBUGENABLED // Path vertex type
			if (ptc.debugEnabled)
			{
				float3 DebugP = ptc.P;
				float3 color = float3(1.0, 0.0, 0.0);
				float size = 25.0f;
				bool doPrint = true;
				if (!hasCollision)
				{
					// Big trouble so big red cross
				}
				else
				{
					if (ptc.lastSurfaceIntersection == D_INTERSECTION_MEDIUM)
					{
						if (ScatteringType == D_SCATT_TYPE_MIE || ScatteringType == D_SCATT_TYPE_RAY)
						{
							color = float3(1.0, 1.0, 0.0);
							size = 2.0;
					//		doPrint = false;
						}
						else if (ScatteringType == D_SCATT_TYPE_ABS)
						{
							// This can happen when we get out of the volume during integration
							color = float3(0.25, 0.25, 0.25);
							size = 2.0;
					//		doPrint = false;
						}
					}
					else
					{
						if (ptc.lastSurfaceIntersection == D_INTERSECTION_GROUND)
						{
							DebugP = nextRay.o;
							color = float3(204, 102, 0) / 255.0f;
							size = 5.0;
						}
						else if (ptc.lastSurfaceIntersection == D_INTERSECTION_NULL)
						{
							color = float3(0.5, 0.5, 1.0);
							size = 2.0;
						}
					}
				}
				if(doPrint)
					addGpuDebugCross(ToDebugWorld + DebugP, color, 2.0);
			}
#endif

            if (hasCollision && ptc.lastSurfaceIntersection == D_INTERSECTION_MEDIUM)
            {
                float lightL;
                float bsdfL;
                float beamTransmittance;
                float lightPdf, bsdfPdf;
                float misWeight;
                float3 sampleDirection;
                bool isDeltaLight;

                // What we do here: (1) sample light, (2) apply phase, (3) update throughput, No MIS, matches PBRT perfectly
                lightGenerateSample(ptc, sampleDirection, lightL, lightPdf, beamTransmittance, isDeltaLight);
                phaseEvaluateSample(ptc, sampleDirection, ScatteringType, bsdfL, bsdfPdf);

#if MULTISCATAPPROX_ENABLED
                // Trying some approximation to multi scattering
                const float globalL = transmittance * weight * (lightL) / (lightPdf);
                // We do not apply beamTransmittance here because otherwise no multi scatter in shadowed region. But now strong forward mie scattering can leak through opaque.

                float multiScatteredLuminance = 0.0f; // Absorption
                if (ScatteringType == D_SCATT_TYPE_MIE || ScatteringType == D_SCATT_TYPE_RAY)
                {
                    MediumSample medium = sampleMedium(ptc);

                    float viewHeight = length(ptc.P);
                    const float3 UpVector = ptc.P / viewHeight;
                    float SunZenithCosAngle = dot(sun_direction, UpVector);

                    multiScatteredLuminance += dot(ptc.wavelengthMask,
                                                   GetMultipleScattering(
                                                       ptc.Atmosphere, medium.scattering, medium.extinction, ptc.P,
                                                       SunZenithCosAngle));
                }

                // multiScatteredLuminance is the integral over the sphere of the incoming luminance assuming a uniform phase function at current point. 
                // The phase and scattering probability is part of the LUT alread. So we do not multiply with bsdfL as this is for the directional sun only.
                Lv += globalL * ((beamTransmittance * bsdfL) + multiScatteredLuminance);
                L += throughput * Lv;
                throughput *= transmittance;

                if (ptc.opaqueHit)
                {
                    //diffuse
                    float NdotL = max(dot(ptc.N, sunDir), 0.0f); // Lambertian diffuse part
                    //float viewHeight = length(ptc.P);
                    //const float3 UpVector = ptc.P / viewHeight;
                    //const float NdotL = saturate(dot(ptc.N, sunDir));

                    const float albedo = 0.5f;
                    float LightRadiance = 1.0f; //light_radiance();  // Wavelength-based light radiance
                    const float DiffuseEval = albedo * (1.0f / PI);
                    DiffuseReturnValue = albedo * LightRadiance * NdotL;
                }


                hasScattered = true;
                break;
#else

				Lv += lightL * bsdfL * beamTransmittance / (lightPdf);
				L += weight * throughput * Lv * transmittance;
				throughput *= transmittance;
#endif

                if (insideAnyVolume(ptc, nextRay))
                {
                    hasScattered = true;
                    ptc.singleScatteringRay = false;
#if MIE_PHASE_IMPORTANCE_SAMPLING
					// This code is also valid for the uniform sampling, but we optimise it out if we do not use anisotropic phase importance sampling.
					float phaseValue, phasePdf;
					phaseGenerateSample(ptc, nextRay.d, ScatteringType, phaseValue, phasePdf);
					throughput *= phaseValue / phasePdf;
#else
                    nextRay.d = getUniformSphereSample(random01(ptc), random01(ptc)); // Simple uniform distribution.
#endif
                }
            }
            //else if (insideAnyVolume(ptc, nextRay))	// Not needed because no internal acceleration structure
            //{
            ////	step--;	// to not have internal subdivision affect path depth
            //	break;
            //}
            else if (ptc.lastSurfaceIntersection == D_INTERSECTION_NULL || (ptc.lastSurfaceIntersection ==
                D_INTERSECTION_GROUND && !GROUND_GI_ENABLED))
            {
                // No intersection within the range
                //return ptc.wavelengthMask.g*0.5;
                break;
            }
        }
        else
        {
#if DEBUGENABLED // Unhandled/wrong Path vertex
			if (ptc.debugEnabled)
			{
				float3 DebugP = ToDebugWorld + ptc.P;
				addGpuDebugLine(DebugP, DebugP + float3(0, 15, 0), float3(1.0, 0.0, 0.0));
			}
#endif
            // Exit as we have exited the atmosphere volume
            break;
        }

        ray = nextRay;
        step++;
    }

    if (!hasScattered && !ptc.opaqueHit && step == 0)
    {
        L += throughput * dot(ptc.wavelengthMask, GetSunLuminance(camPos, WorldDir, ptc.Atmosphere.BottomRadius));
    }

    ptc.transmittance = !hasScattered ? throughput : 0.0f;

    return float2(L, DiffuseReturnValue);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


float4 WaveLengthToRGB(float Wavelength)
{
    float Red, Green, Blue;
    float factor;

    if ((Wavelength >= 380) && (Wavelength < 440))
    {
        Red = -(Wavelength - 440) / (440 - 380);
        Green = 0.0;
        Blue = 1.0;
    }
    else if ((Wavelength >= 440) && (Wavelength < 490))
    {
        Red = 0.0;
        Green = (Wavelength - 440) / (490 - 440);
        Blue = 1.0;
    }
    else if ((Wavelength >= 490) && (Wavelength < 510))
    {
        Red = 0.0;
        Green = 1.0;
        Blue = -(Wavelength - 510) / (510 - 490);
    }
    else if ((Wavelength >= 510) && (Wavelength < 580))
    {
        Red = (Wavelength - 510) / (580 - 510);
        Green = 1.0;
        Blue = 0.0;
    }
    else if ((Wavelength >= 580) && (Wavelength < 645))
    {
        Red = 1.0;
        Green = -(Wavelength - 645) / (645 - 580);
        Blue = 0.0;
    }
    else if ((Wavelength >= 645) && (Wavelength < 781))
    {
        Red = 1.0;
        Green = 0.0;
        Blue = 0.0;
    }
    else
    {
        Red = 0.0;
        Green = 0.0;
        Blue = 0.0;
    }
    // Let the intensity fall off near the vision limits
    if ((Wavelength >= 380) && (Wavelength < 420))
    {
        factor = 0.3 + 0.7 * (Wavelength - 380) / (420 - 380);
    }
    else if ((Wavelength >= 420) && (Wavelength < 701))
    {
        factor = 1.0;
    }
    else if ((Wavelength >= 701) && (Wavelength < 781))
    {
        factor = 0.3 + 0.7 * (780 - Wavelength) / (780 - 700);
    }
    else
    {
        factor = 0.0;
    }

    const float IntensityMax = 1.0;
    const float Gamma = 0.8;
    // Don't want 0^x = 1 for x <> 0
    float RedIntensity = Red == 0.0 ? 0 : saturate(IntensityMax * pow(Red * factor, Gamma));
    float GreenIntensity = Green == 0.0 ? 0 : saturate(IntensityMax * pow(Green * factor, Gamma));
    float BlueIntensity = Blue == 0.0 ? 0 : saturate(IntensityMax * pow(Blue * factor, Gamma));

    return float4(RedIntensity, GreenIntensity, BlueIntensity, 1.0);
}

struct PixelOutputStruct
{
    float4 Luminance : SV_TARGET0;
#if GAMEMODE_ENABLED==0
    float4 Transmittance : SV_TARGET1;
    float4 Diffuse : SV_TARGET2;
#endif
};

PixelOutputStruct RenderPathTracingPS(VertexOutput Input)
{
    float2 pixPos = Input.position.xy;

    const float NumSample = 1.0f;
    float3 OutputLuminance = 0.0f;
    float3 OutputDiffuse = 0.0f;
    float3 ScatteringMajorant = rayleigh_scattering + mie_scattering;
    float3 ExtinctionMajorant = rayleigh_scattering + mie_extinction + absorption_extinction;

    PathTracingContext ptc = (PathTracingContext)0;
    ptc.Atmosphere = GetAtmosphereParameters();
    ptc.extinctionMajorant = mean(ExtinctionMajorant);
    ptc.scatteringMajorant = mean(ScatteringMajorant);
    ptc.lastSurfaceIntersection == D_INTERSECTION_MEDIUM;
    ptc.P = 0.0f;
    ptc.V = 0.0f;
    ptc.ray = createRay(0.0f, 0.0f);
    ptc.screenPixelPos = pixPos;
    ptc.randomState = (pixPos.x + pixPos.y * float(gResolution.x)) + uint(gFrameId * 123u) % 32768u;
    ptc.singleScatteringRay = true;
    ptc.opaqueHit = false;
    ptc.transmittance = 1.0f; // initialise to 1 for above atmosphere transmittance to be 1..
    ptc.debugEnabled = all(ptc.screenPixelPos == gMouseLastDownPos);
    ptc.lastDiffrence = 0;


    //ptc.debugEnabled = ptc.screenPixelPos.x % 512 ==0 && ptc.screenPixelPos.y % 512 == 0;

#if 0
	float zeta = random01(ptc);
#else
    // Using bluenoise as a first scramble for wavelength selection for delta tracking
    const float noise = BlueNoise2dTexture[(pixPos * 7) % 64].r;
    float zeta = noise + (gFrameId % 3) / 3.0;
    zeta = zeta > 1.0f ? zeta - 1.0f : zeta;
#endif


    float3 mask = 0.0f;
    if (zeta < 1.0 / 3.0)
    {
        ptc.extinctionMajorant = ExtinctionMajorant.r;
        ptc.scatteringMajorant = ScatteringMajorant.r;
        ptc.wavelengthMask = float3(1.0, 0.0, 0.0);
    }
    else if (zeta < 2.0 / 3.0)
    {
        ptc.extinctionMajorant = ExtinctionMajorant.g;
        ptc.scatteringMajorant = ScatteringMajorant.g;
        ptc.wavelengthMask = float3(0.0, 1.0, 0.0);
    }
    else
    {
        ptc.extinctionMajorant = ExtinctionMajorant.b;
        ptc.scatteringMajorant = ScatteringMajorant.b;
        ptc.wavelengthMask = float3(0.0, 0.0, 1.0);
    }


    float wavelengthPdf = 1.0 / 3.0;
    const float3 wavelengthWeight = ptc.wavelengthMask / wavelengthPdf;
    float2 result = LightIntegratorInner(Input, ptc);
    OutputLuminance = result.x;
    OutputDiffuse = result.y;

    float3 rgbDiffuse = OutputDiffuse; //  WaveLengthToRGB(OutputDiffuse);

    PixelOutputStruct output;

    //if (pixPos.x < 512 && pixPos.y < 512)
    //{
    //	output.Luminance = float4(0.2*SkyViewLutTexture.SampleLevel(samplerLinearClamp, pixPos / float2(512, 512), 0).rgb, 1.0);
    //	output.Transmittance = float4(0,0,0,1);
    //	return output;
    //}

#if GAMEMODE_ENABLED
	output.Luminance = float4(OutputLuminance   * wavelengthWeight, dot(ptc.transmittance * wavelengthWeight, float3(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f)));
#else
    output.Luminance = float4(OutputLuminance * wavelengthWeight, 1.0f);
    output.Transmittance = ptc.opaqueHit ? float4(1.0f, 1.0f, 1.0f, 1.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    output.Diffuse = float4(rgbDiffuse.x, rgbDiffuse.y, rgbDiffuse.z, 1.0f);
#endif
    return output;
}
