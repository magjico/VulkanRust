#version 450

const float PI = 3.141592653589793;

struct Light {
	vec4 position;
	vec4 color;
};

// check uniform.rs inside render to understand UBO.
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;

    vec4 camPos;
    float exposure;
    float gamma;
    float prefilteredCubMipLevels;
    float scaleIBLAmbient;
} ubo;

layout(set = 0, binding = 1) readonly buffer Lights {
	Light lights[];
};

// Take a look at create_material_descriptor_set_layout in shader.rs
layout(set = 1, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 1, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 1, binding = 2) uniform sampler2D normalTexture;
layout(set = 1, binding = 3) uniform sampler2D occlusionTexture;
layout(set = 1, binding = 4) uniform sampler2D emissiveTexture;

layout(push_constant) uniform PushConstants {
    layout(offset = 68)	 float metallicFactor;	// How metallic the surface is							(offset 68)
    float roughnessFactor;						// How rough the surface is								(offset 72)
	float _padding;								// Padding to align baseColorFactor						(offset 76)
	vec4 baseColorFactor;						// RGB base color and alpha								(offset 80)
    int baseColorTextureSet;					// Texture coordinate set for base color				(offset 96)
    int physicalDescriptorTextureSet;			// Texture coordinate set for metallic-roughness		(offset 100)
    int normalTextureSet;						// Texture coordinate set for normal map				(offset 104)
    int occlusionTextureSet;					// Texture coordinate set for occlusion					(offset 108)
    int emissiveTextureSet;						// Texture coordinate set for emission					(offset 112)
	float alphaMask;							// Whether to use alpha masking							(offset 116)
    float alphaMaskCutoff;						// Alpha threshold for masking							(offset 120)
	// Total: 124 bytes / 128 bytes used.
} pcs;

// Take a look at the out vector inside shader.vert
layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal; 
layout(location = 2) in vec2 fragUV0;
layout(location = 3) in vec2 fragUV1;
layout(location = 4) in vec4 fragTangent;

layout(location = 0) out vec4 outColor;

//===============================================
// Texture function
//===============================================

vec2 uvFor(int set) {
	return set == 0 ? fragUV0 : fragUV1;
}

//===============================================
// BSDF - PBR Distribution functions
//===============================================

/*
	* Compute the H oriented microfacets proportion.
	*
	* Disney implementation of the GGX distribution.
	* Normally the GGX distribution only use a squared roughness, but Disney find that
	* using the squared of the squared roughness result into a more perceptual linear progression
	* of the roughness for the human eye.
	*
	* This is also a simplified version of the GGX Distribution which assume that:
	* 1) N and H are normalized vectors.
	* 2) roughness is the same following the X and Y principal axis.
	*
	* @param NdotH : dot(Normal, HalfVector)
	* @param roughness : material roughness [0, 1]
	* @return: normal density value
*/
float disneyGGXDistribution(float NdotH, float roughness)
{
    float alpha = roughness * roughness;
	float alpha_disney = alpha * alpha;
	float NdotH2 = NdotH * NdotH;

	float denom = (NdotH2 * (alpha_disney - 1.0) + 1.0);

	return alpha_disney / denom;
}


/*
	* Schlick optimisation of the Smith Shadowing Function.
	*
	* @param NdotV : dot(Normal, View)
	* @param NdotL : dot(Normal, LumensDirection)
	* @param k : schlick factor
	* @return microfacets proportion.
*/
float schlickOptimisation(float NdotV, float NdotL, float k)
{
	float masking = NdotV / (NdotV * (1.0 - k) + k);
	float shadowing = NdotL / (NdotL * (1.0 - k) + k);

	return masking * shadowing;
}

/*
	* Compute the proportion of microfacets which are not occulded.
	*
	* This is the Schlick optimisation / approximation for ponctual light.
	*
	* Same as the GGX distribution it assume that:
	* 1) All vectors are normalized
	* 2) roughness is the same following the X and Y principal axis.
	*
	* @param NdotV : dot(Normal, View)
	* @param NdotL : dot(Normal, LumensDirection)
	* @param roughness: material roughness [0, 1]
	* @return microfacets proportion.
*/
float smithPonctualShadowingFunction(float NdotV, float NdotL, float roughness)
{
	float r = roughness + 1.0;
	float k = (r * r) / 8.0;

	return schlickOptimisation(NdotV, NdotL, k);
}

/*
	* Compute the proportion of microfacets which are not occulded.
	*
	* This is the Schlick optimisation / approximation for IBL (Image Based Lightning).
	*
	* Same as the GGX distribution it assume that:
	* 1) All vectors are normalized
	* 2) roughness is the same following the X and Y principal axis.
	*
	* @param NdotV : dot(Normal, View)
	* @param NdotL : dot(Normal, LumensDirection)
	* @param roughness: material roughness [0, 1]
	* @return microfacets proportion.
*/
float smithIBLShadowingFunction(float NdotV, float NdotL, float roughness)
{
	float k = (roughness * roughness) / 2.0;

	return schlickOptimisation(NdotV, NdotL, k);
}

/*
	* Compute Fresnel reflexivity with Schlick approximation.
	* Which proportion of the light is reflected vs refracted for a given view angle.
	*
	* LIMITATION: For complex metals such as beryllium,
	* Schlick’s approximation fails to accurately reproduce
	* the actual Fresnel curve.
	*
	* @param cosTheta : cosinus of a view angle
	* @param F0: derived from the material’s index of refraction (IOR)
	* @return the Fresnel reflectivity for each RGB channel at the given angle cosTheta.
*/
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

/*
	* ACES tonemapping — for PBR
*/
vec3 ACESFilm(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

//===============================================
// Main
//===============================================

void main()
{
	// 1 - sample material textures
    vec4 baseColor = texture(baseColorTexture, uvFor(pcs.baseColorTextureSet)) * pcs.baseColorFactor;

	// alpha-masking
	if (pcs.alphaMask == 1.0 && baseColor.a < pcs.alphaMaskCutoff) discard;

	vec2 metallicRoughness = texture(metallicRoughnessTexture, uvFor(pcs.physicalDescriptorTextureSet)).bg;
	float metallic	= metallicRoughness.x * pcs.metallicFactor;
	float roughness	= metallicRoughness.y * pcs.roughnessFactor;
	float ao		= texture(occlusionTexture, uvFor(pcs.occlusionTextureSet)).r;
	vec3 emissive	= texture(emissiveTexture, uvFor(pcs.emissiveTextureSet)).rgb;

	// 2 - Calculate Normal in tangent space
	vec3 N = normalize(fragNormal);
	if (pcs.normalTextureSet >= 0) {
		vec3 tangentNormal = texture(normalTexture, uvFor(pcs.normalTextureSet)).xyz * 2.0 - 1.0;
		vec3 T = normalize(fragTangent.xyz);
		vec3 B = normalize(cross(N, T)) * fragTangent.w;
		mat3 TBN = mat3(T, B, N);
		N = normalize(TBN * tangentNormal);
	}

	// 3 - Calculate View vector
	vec3 V = normalize(ubo.camPos.xyz - fragWorldPos);

	// 4 - Calculate F0 (base reflexivity)
	vec3 F0 = vec3(0.04); // 0.04 is the normal-incidence reflectivity of dielectrics (non-metallic materials) — corresponds to a refractive index (IOR) of 1.5, the average value for glass and plastic.
	F0 = mix(F0, baseColor.rgb, metallic);

	// 5 - Calculate lighting for each light
	// Initialize lightning
	vec3 Lo = vec3(0.0);

	for (int i=0; i < lights.length(); i++) {
		vec3 lightPos = lights[i].position.xyz;
		vec3 lightColor = lights[i].color.rgb * lights[i].color.w;

		// Calculate light direction and distance
		vec3 L				= normalize(lightPos - fragWorldPos);
		float dist			= length(lightPos - fragWorldPos);
		float attenuation 	= 1.0 / (dist * dist);
		vec3 radiance		= lightColor * attenuation;

		// Calculate hald-vector (the normalized vector halfway between view and light direction)
		vec3 H = normalize(V + L);

		// Calculate BRDF terms
		float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

		// Specular BRDF
        float D = disneyGGXDistribution(NdotH, roughness);
        float G = smithPonctualShadowingFunction(NdotV, NdotL, roughness);
        vec3 F = fresnelSchlick(HdotV, F0);

		vec3 specular = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);

		// Energy conservation
		vec3 kD = vec3(1.0) - F;
		kD *= 1.0 - metallic;

		// Add to outgoing radiance
		Lo += (kD * baseColor.rgb / PI + specular) * radiance * NdotL;
	}

	vec3 ambient	= vec3(0.03) * baseColor.rgb * ao;
	vec3 color		= ambient + Lo + emissive;
	// color = color / (color + vec3(1.0));
	color = ACESFilm(color * ubo.exposure);
	color = pow(color, vec3(1.0 / ubo.gamma));

    outColor = vec4(color, baseColor.a);
}