// Copyright Epic Games, Inc. All Rights Reserved.


#include "./Resources/Common.hlsl"


Texture2D<float4> PathtracingLuminanceTexture				: register(t2);
Texture2D<float4> PathtracingTransmittanceTexture			: register(t3);
Texture2D<float4> PathtracingDiffuseTexture			        : register(t4);

float sRGB(float x)
{
	if (x <= 0.00031308)
		return 12.92 * x;
	else
		return 1.055*pow(x, (1.0 / 2.4)) - 0.055;
}

float4 sRGB(float4 vec)
{
	return float4(sRGB(vec.x), sRGB(vec.y), sRGB(vec.z), vec.w);
}





struct PixelOutputStruct
{
	float4 HdrBuffer		: SV_TARGET0;
	float4 Transmittance	: SV_TARGET1;
};

float4 PreCookTexture(float4 original_texture, float3 white_point, float exposure)
{
	float3 original_rgb = original_texture.rgb;

	// Step 1: Reverse gamma correction
	float3 inverted = pow(original_rgb, 2.2);

	// Step 2: Reverse inversion
	float3 attenuated = 1.0 - inverted;

	// Step 3: Reverse exponential decay
	float3 scaled_rgb = -log(attenuated);

	// Step 4: Reverse exposure and white point scaling
	float3 pre_cooked_rgb = scaled_rgb * white_point / exposure;

	return float4(pre_cooked_rgb, original_texture.a);
}

PixelOutputStruct ApplySkyAtmospherePS(VertexOutput input)
{
	uint2 texCoord = input.position.xy;

	float4 PathtracingLuminance		= PathtracingLuminanceTexture.Load(uint3(texCoord, 0));
	float4 PathtracingTransmittance	= PathtracingTransmittanceTexture.Load(uint3(texCoord, 0));
	float4 PathtracingDiffuse	= PathtracingDiffuseTexture.Load(uint3(texCoord, 0));

	PathtracingLuminance = PathtracingLuminance.w > 0.0 ? PathtracingLuminance / PathtracingLuminance.w : float4(0.0, 0.0, 0.0, 1.0);
	PathtracingTransmittance = PathtracingTransmittance;
	PathtracingDiffuse = PathtracingDiffuse.w > 0.0 ? PathtracingDiffuse / PathtracingDiffuse.w : float4(0.0, 0.0, 0.0, 0.0);
	PathtracingDiffuse = PreCookTexture(PathtracingDiffuse, float3(1.08241, 0.96756, 0.95003), 10.0);
	float4 blendedColor = float4(PathtracingTransmittance.w * PathtracingLuminance.xyz + (1.0 - PathtracingTransmittance.w) * PathtracingDiffuse.xyz, 1.0);
	PixelOutputStruct output;
	output.HdrBuffer = PathtracingTransmittance.x > 0.1f ? PathtracingDiffuse:  PathtracingLuminance;
	output.Transmittance = float4(0,0,0,0);
	return output;
}


float4 PostProcessPS(VertexOutput input) : SV_TARGET
{
	uint2 texCoord = input.position.xy;

	float4 rgbA = texture2d.Load(uint3(texCoord,0));
	rgbA /= rgbA.aaaa;	// Normalise according to sample count when path tracing

	// Similar setup to the Bruneton demo
	float3 white_point = float3(1.08241, 0.96756, 0.95003);
	float exposure = 10.0;
	return float4( pow((float3) 1.0 - exp(-rgbA.rgb / white_point * exposure), (float3)(1.0 / 2.2)), 1.0 );
}



