/*
* Vulkan Example - Compute shader N-body simulation using two passes and shared compute shader memory
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false
#if defined(__ANDROID__)
// Lower particle count on Android for performance reasons
#define PARTICLES_PER_ATTRACTOR 3 * 1024
#else
#define PARTICLES_PER_ATTRACTOR 24 * 1024
#endif

class VulkanExample : public VulkanExampleBase
{
public:
	uint32_t numParticles;

	struct {
		vks::Texture2D particle;
		vks::Texture2D gradient;
	} textures;

	struct {
		VkPipelineVertexInputStateCreateInfo inputState;
		std::vector<VkVertexInputBindingDescription> bindingDescriptions;
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	// Resources for the graphics part of the example
	struct {
		uint32_t queueFamilyIndex;					// Used to check if compute and graphics queue families differ and require additional barriers
		vks::Buffer uniformBuffer;					// Contains scene matrices
		VkDescriptorSetLayout descriptorSetLayout;	// Particle system rendering shader binding layout
		VkDescriptorSet descriptorSet;				// Particle system rendering shader bindings
		VkPipelineLayout pipelineLayout;			// Layout of the graphics pipeline
		VkPipeline pipeline;						// Particle rendering pipeline
		VkSemaphore semaphore;                      // Execution dependency between compute & graphic submission
		struct {
			glm::mat4 projection;
			glm::mat4 view;
			glm::vec2 screenDim;
		} ubo;
	} graphics;

	// Resources for the compute part of the example
	struct {
		uint32_t queueFamilyIndex;					// Used to check if compute and graphics queue families differ and require additional barriers
		vks::Buffer storageBuffer;					// (Shader) storage buffer object containing the particles
		vks::Buffer uniformBuffer;					// Uniform buffer object containing particle system parameters
		VkQueue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		VkCommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		VkCommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
		VkSemaphore semaphore;                      // Execution dependency between compute & graphic submission
		VkDescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout
		VkDescriptorSet descriptorSet;				// Compute shader bindings
		VkPipelineLayout pipelineLayout;			// Layout of the compute pipeline
		VkPipeline pipelineCalculate;				// Compute pipeline for N-Body velocity calculation (1st pass)
		VkPipeline pipelineIntegrate;				// Compute pipeline for euler integration (2nd pass)
		VkPipeline blur;
		VkPipelineLayout pipelineLayoutBlur;
		VkDescriptorSetLayout descriptorSetLayoutBlur;
		VkDescriptorSet descriptorSetBlur;
		struct computeUBO {							// Compute shader uniform block object
			float deltaT;							//		Frame delta time
			int32_t particleCount;
		} ubo;
	} compute;

	// SSBO particle declaration
	struct Particle {
		glm::vec4 pos;								// xyz = position, w = mass
		glm::vec4 vel;								// xyz = velocity, w = gradient texture position
	};

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "E3 meta-space physics";
		settings.overlay = false;
		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(-26.0f, 75.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -3.0f));
		camera.movementSpeed = 0.3f;
	}

	~VulkanExample()
	{
		// Graphics
		graphics.uniformBuffer.destroy();
		vkDestroyPipeline(device, graphics.pipeline, nullptr);
		vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout, nullptr);
		vkDestroySemaphore(device, graphics.semaphore, nullptr);

		// Compute
		compute.storageBuffer.destroy();
		compute.uniformBuffer.destroy();
		vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
		vkDestroyPipeline(device, compute.pipelineCalculate, nullptr);
		vkDestroyPipeline(device, compute.pipelineIntegrate, nullptr);
		vkDestroySemaphore(device, compute.semaphore, nullptr);
		vkDestroyCommandPool(device, compute.commandPool, nullptr);

		textures.particle.destroy();
		textures.gradient.destroy();
	}

	void loadAssets()
    
	{
		textures.particle.loadFromFile(getAssetPath() + "textures/particle01_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
		textures.gradient.loadFromFile(getAssetPath() + "textures/particle_gradient_rgba.ktx", VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			// Acquire barrier
			if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
			{
				VkBufferMemoryBarrier buffer_barrier =
				{
					VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
					nullptr,
					0,
					VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
					compute.queueFamilyIndex,
					graphics.queueFamilyIndex,
					compute.storageBuffer.buffer,
					0,
					compute.storageBuffer.size
				};

				vkCmdPipelineBarrier(
					drawCmdBuffers[i],
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
					0,
					0, nullptr,
					1, &buffer_barrier,
					0, nullptr);
			}

			// Draw the particle system using the update vertex buffer
			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSet, 0, nullptr);

			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, &compute.storageBuffer.buffer, offsets);
			vkCmdDraw(drawCmdBuffers[i], numParticles, 1, 0, 0);

			drawUI(drawCmdBuffers[i]);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			// Release barrier
			if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
			{
				VkBufferMemoryBarrier buffer_barrier =
				{
					VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
					nullptr,
					VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
					0,
					graphics.queueFamilyIndex,
					compute.queueFamilyIndex,
					compute.storageBuffer.buffer,
					0,
					compute.storageBuffer.size
				};

				vkCmdPipelineBarrier(
					drawCmdBuffers[i],
					VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0,
					0, nullptr,
					1, &buffer_barrier,
					0, nullptr);
			}

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}

	}

	void buildComputeCommandBuffer() // some setup of compute stages
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));

		// Acquire barrier
		if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
		{
			VkBufferMemoryBarrier buffer_barrier =
			{
				VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				nullptr,
				0,
				VK_ACCESS_SHADER_WRITE_BIT,
				graphics.queueFamilyIndex,
				compute.queueFamilyIndex,
				compute.storageBuffer.buffer,
				0,
				compute.storageBuffer.size
			};

			vkCmdPipelineBarrier(
				compute.commandBuffer,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				0, nullptr,
				1, &buffer_barrier,
				0, nullptr);
		}

		// First pass: Calculate particle movement
		// -------------------------------------------------------------------------------------------------------
		vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineCalculate);
		vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);
		vkCmdDispatch(compute.commandBuffer, numParticles / 256, 1, 1);

		// Add memory barrier to ensure that the computer shader has finished writing to the buffer
		VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		bufferBarrier.buffer = compute.storageBuffer.buffer;
		bufferBarrier.size = compute.storageBuffer.descriptor.range;
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		// Transfer ownership if compute and graphics queue family indices differ
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		vkCmdPipelineBarrier(
			compute.commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_FLAGS_NONE,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr);

		// Second pass: Integrate particles
		// -------------------------------------------------------------------------------------------------------
		//vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineIntegrate);
		//vkCmdDispatch(compute.commandBuffer, numParticles / 256, 1, 1);

		// Release barrier
		if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
		{
			VkBufferMemoryBarrier buffer_barrier =
			{
				VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				nullptr,
				VK_ACCESS_SHADER_WRITE_BIT,
				0,
				compute.queueFamilyIndex,
				graphics.queueFamilyIndex,
				compute.storageBuffer.buffer,
				0,
				compute.storageBuffer.size
			};

			vkCmdPipelineBarrier(
				compute.commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				0,
				0, nullptr,
				1, &buffer_barrier,
				0, nullptr);
		}

		vkEndCommandBuffer(compute.commandBuffer);
	}
    
    static float random(float min, float max)
    {
        return (min + 1) + (((float) rand()) / (float) RAND_MAX) * (max - (min + 1));
    }
    
    static glm::vec4 random_quaternion() {
      float x,y,z, u,v,w, s;
      do { x = random(-1.0f, 1.0f); y = random(-1.0f, 1.0f); z = x*x + y*y; } while (z > 1.0f);
      do { u = random(-1.0f, 1.0f); v = random(-1.0f, 1.0f); w = u*u + v*v; } while (w > 1.0f);
      s = sqrt((1.0f-z) / w);
      return glm::vec4(y, s*u, s*v, x);
    }
    

	// Setup and fill the compute shader storage buffers containing the particles
	void prepareStorageBuffers()
	{
        numParticles = PARTICLES_PER_ATTRACTOR;
        std::vector<Particle> particleBuffer(numParticles);

        std::default_random_engine rndEngine(benchmark.active ? 0 : (unsigned)time(nullptr));
        std::uniform_real_distribution<float> rndDist(-2.0f, 2.0f);
		std::uniform_real_distribution<float> rndAngle(0.0f, 6.28318530718f);

        for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; j++)
        {
            Particle &particle = particleBuffer[j];
            particle.pos = glm::vec4(glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine)), 10.0f);
			glm::vec3 orient(glm::normalize(glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine))));
            particle.vel = glm::vec4(orient, 0.0f);
        }

		/*for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; j++)
        {
            Particle &particle = particleBuffer[j];
            particle.pos = glm::vec4(glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine)), 10.0f);
			glm::vec3 orient(glm::normalize(glm::vec3(rndAngle(rndEngine), rndAngle(rndEngine), rndDist(rndEngine))));
            particle.vel = glm::vec4(orient, 0.0f);
        }*/
        
        {
            compute.ubo.particleCount = numParticles;
       
            VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);
            
            // Staging
            // SSBO won't be changed on the host after upload so copy to device local memory

            vks::Buffer stagingBuffer;

            vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &stagingBuffer,
                storageBufferSize,
                particleBuffer.data());

            vulkanDevice->createBuffer(
                // The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                &compute.storageBuffer,
                storageBufferSize);

            // Copy from staging buffer to storage buffer
            VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            VkBufferCopy copyRegion = {};
            copyRegion.size = storageBufferSize;
            vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, compute.storageBuffer.buffer, 1, &copyRegion);
            // Execute a transfer barrier to the compute queue, if necessary
            if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
            {
                VkBufferMemoryBarrier buffer_barrier =
                {
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    nullptr,
                    VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                    0,
                    graphics.queueFamilyIndex,
                    compute.queueFamilyIndex,
                    compute.storageBuffer.buffer,
                    0,
                    compute.storageBuffer.size
                };

                vkCmdPipelineBarrier(
                    copyCmd,
                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0, nullptr,
                    1, &buffer_barrier,
                    0, nullptr);
            }
            vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

            stagingBuffer.destroy();

            // Binding description
            vertices.bindingDescriptions.resize(1);
            vertices.bindingDescriptions[0] =
                vks::initializers::vertexInputBindingDescription(
                    VERTEX_BUFFER_BIND_ID,
                    sizeof(Particle),
                    VK_VERTEX_INPUT_RATE_VERTEX);

            // Attribute descriptions
            // Describes memory layout and shader positions
            vertices.attributeDescriptions.resize(2);
            // Location 0 : Position
            vertices.attributeDescriptions[0] =
                vks::initializers::vertexInputAttributeDescription(
                    VERTEX_BUFFER_BIND_ID,
                    0,
                    VK_FORMAT_R32G32B32A32_SFLOAT,
                    offsetof(Particle, pos));
            // Location 1 : Velocity (used for gradient lookup)
            vertices.attributeDescriptions[1] =
                vks::initializers::vertexInputAttributeDescription(
                    VERTEX_BUFFER_BIND_ID,
                    1,
                    VK_FORMAT_R32G32B32A32_SFLOAT,
                    offsetof(Particle, vel));

            // Assign to vertex buffer
            vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
            vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
            vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
            vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
            vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
        }
	}

	void setupDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2)
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				2);

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void setupDescriptorSetLayout()
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 2),
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				static_cast<uint32_t>(setLayoutBindings.size()));

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &graphics.descriptorSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&graphics.descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &graphics.pipelineLayout));
	}

	void setupDescriptorSet()
	{
		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&graphics.descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSet));

		std::vector<VkWriteDescriptorSet> writeDescriptorSets;
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(graphics.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &textures.particle.descriptor),
			vks::initializers::writeDescriptorSet(graphics.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &textures.gradient.descriptor),
			vks::initializers::writeDescriptorSet(graphics.descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &graphics.uniformBuffer.descriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}

	void preparePipelines()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
				0,
				VK_FALSE);

		VkPipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				VK_POLYGON_MODE_FILL,
				VK_CULL_MODE_NONE,
				VK_FRONT_FACE_COUNTER_CLOCKWISE,
				0);

		VkPipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				0xf,
				VK_FALSE);

		VkPipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

		VkPipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_FALSE,
				VK_FALSE,
				VK_COMPARE_OP_ALWAYS);

		VkPipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		VkPipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(
				VK_SAMPLE_COUNT_1_BIT,
				0);

		std::vector<VkDynamicState> dynamicStateEnables = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				static_cast<uint32_t>(dynamicStateEnables.size()),
				0);

		// Rendering pipeline
		// Load shaders
		std::array<VkPipelineShaderStageCreateInfo,2> shaderStages;

		shaderStages[0] = loadShader(getShadersPath() + "computenbody/particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "computenbody/particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		VkGraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				graphics.pipelineLayout,
				renderPass,
				0);

		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = renderPass;

		// Additive blending
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &graphics.pipeline));
	}

	void prepareGraphics()
	{
		prepareStorageBuffers();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorSet();

		// Semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &graphics.semaphore));
	}

	void prepareCompute()
	{
		// Create a compute capable device queue
		// The VulkanDevice::createLogicalDevice functions finds a compute capable queue and prefers queue families that only support compute
		// Depending on the implementation this may result in different queue family indices for graphics and computes,
		// requiring proper synchronization (see the memory barriers in buildComputeCommandBuffer)
		vkGetDeviceQueue(device, compute.queueFamilyIndex, 0, &compute.queue);

		// Create compute pipeline
		// Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Particle position storage buffer
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				VK_SHADER_STAGE_COMPUTE_BIT,
				0),
			// Binding 1 : Uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				VK_SHADER_STAGE_COMPUTE_BIT,
				1),
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				static_cast<uint32_t>(setLayoutBindings.size()));

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device,	&descriptorLayout, nullptr,	&compute.descriptorSetLayout));

		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&compute.descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr,	&compute.pipelineLayout));

		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&compute.descriptorSetLayout,
				1);

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));

		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
		{
			// Binding 0 : Particle position storage buffer
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				0,
				&compute.storageBuffer.descriptor),
			// Binding 1 : Uniform buffer
			vks::initializers::writeDescriptorSet(
				compute.descriptorSet,
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				1,
				&compute.uniformBuffer.descriptor)
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);

		// Create pipelines
		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);

		// 1st pass
		computePipelineCreateInfo.stage = loadShader(getShadersPath() + "computenbody/particle_calculate.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

		// Set shader parameters via specialization constants
		struct SpecializationData {
			uint32_t sharedDataSize;
			float gravity;
			float power;
			float soften;
		} specializationData;

		std::vector<VkSpecializationMapEntry> specializationMapEntries;
		specializationMapEntries.push_back(vks::initializers::specializationMapEntry(0, offsetof(SpecializationData, sharedDataSize), sizeof(uint32_t)));
		specializationMapEntries.push_back(vks::initializers::specializationMapEntry(1, offsetof(SpecializationData, gravity), sizeof(float)));
		specializationMapEntries.push_back(vks::initializers::specializationMapEntry(2, offsetof(SpecializationData, power), sizeof(float)));
		specializationMapEntries.push_back(vks::initializers::specializationMapEntry(3, offsetof(SpecializationData, soften), sizeof(float)));

		specializationData.sharedDataSize = std::min((uint32_t)1024, (uint32_t)(vulkanDevice->properties.limits.maxComputeSharedMemorySize / sizeof(glm::vec4)));

		specializationData.gravity = 0.002f;
		specializationData.power = 0.75f;
		specializationData.soften = 0.05f;

		VkSpecializationInfo specializationInfo =
			vks::initializers::specializationInfo(static_cast<uint32_t>(specializationMapEntries.size()), specializationMapEntries.data(), sizeof(specializationData), &specializationData);
		computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipelineCalculate));

		// 2nd pass
		//computePipelineCreateInfo.stage = loadShader(getShadersPath() + "computenbody/particle_integrate.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
		//VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipelineIntegrate));

		// Separate command pool as queue family for compute may be different than graphics
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = compute.queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

		// Create a command buffer for compute operations
		compute.commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, compute.commandPool);

		// Semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore));

		// Signal the semaphore
		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &compute.semaphore;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VK_CHECK_RESULT(vkQueueWaitIdle(queue));

		// Build a single command buffer containing the compute dispatch commands
		buildComputeCommandBuffer();

		// If graphics and compute queue family indices differ, acquire and immediately release the storage buffer, so that the initial acquire from the graphics command buffers are matched up properly
		if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
		{
			// Create a transient command buffer for setting up the initial buffer transfer state
			VkCommandBuffer transferCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, compute.commandPool, true);

			VkBufferMemoryBarrier acquire_buffer_barrier =
			{
				VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				nullptr,
				0,
				VK_ACCESS_SHADER_WRITE_BIT,
				graphics.queueFamilyIndex,
				compute.queueFamilyIndex,
				compute.storageBuffer.buffer,
				0,
				compute.storageBuffer.size
			};
			vkCmdPipelineBarrier(
				transferCmd,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				0, nullptr,
				1, &acquire_buffer_barrier,
				0, nullptr);

			VkBufferMemoryBarrier release_buffer_barrier =
			{
				VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
				nullptr,
				VK_ACCESS_SHADER_WRITE_BIT,
				0,
				compute.queueFamilyIndex,
				graphics.queueFamilyIndex,
				compute.storageBuffer.buffer,
				0,
				compute.storageBuffer.size
			};
			vkCmdPipelineBarrier(
				transferCmd,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
				0,
				0, nullptr,
				1, &release_buffer_barrier,
				0, nullptr);

			vulkanDevice->flushCommandBuffer(transferCmd, compute.queue, compute.commandPool);
		}
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Compute shader uniform buffer block
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&compute.uniformBuffer,
			sizeof(compute.ubo));

		// Map for host access
		VK_CHECK_RESULT(compute.uniformBuffer.map());

		// Vertex shader uniform buffer block
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&graphics.uniformBuffer,
			sizeof(graphics.ubo));

		// Map for host access
		VK_CHECK_RESULT(graphics.uniformBuffer.map());

		updateComputeUniformBuffers();
		updateGraphicsUniformBuffers();
	}

	void updateComputeUniformBuffers()
	{
		compute.ubo.deltaT = paused ? 0.0f : frameTimer * 0.05f;
		memcpy(compute.uniformBuffer.mapped, &compute.ubo, sizeof(compute.ubo));
	}

	void updateGraphicsUniformBuffers()
	{
		graphics.ubo.projection = camera.matrices.perspective;
		graphics.ubo.view = camera.matrices.view;
		graphics.ubo.screenDim = glm::vec2((float)width, (float)height);
		memcpy(graphics.uniformBuffer.mapped, &graphics.ubo, sizeof(graphics.ubo));
	}

	
	// Take a screenshot from the current swapchain image
	// This is done using a blit from the swapchain image to a linear image whose memory content is then saved as a ppm image
	// Getting the image date directly from a swapchain image wouldn't work as they're usually stored in an implementation dependent optimal tiling format
	// Note: This requires the swapchain images to be created with the VK_IMAGE_USAGE_TRANSFER_SRC_BIT flag (see VulkanSwapChain::create)
	void saveScreenshot(const char *filename)
	{
		//screenshotSaved = false;
		bool supportsBlit = true;

		// Check blit support for source and destination
		VkFormatProperties formatProps;

		// Check if the device supports blitting from optimal images (the swapchain images are in optimal format)
		vkGetPhysicalDeviceFormatProperties(physicalDevice, swapChain.colorFormat, &formatProps);
		if (!(formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) {
			std::cerr << "Device does not support blitting from optimal tiled images, using copy instead of blit!" << std::endl;
			supportsBlit = false;
		}

		// Check if the device supports blitting to linear images
		vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, &formatProps);
		if (!(formatProps.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
			std::cerr << "Device does not support blitting to linear tiled images, using copy instead of blit!" << std::endl;
			supportsBlit = false;
		}

		// Source for the copy is the last rendered swapchain image
		VkImage srcImage = swapChain.images[currentBuffer];

		// Create the linear tiled destination image to copy to and to read the memory from
		VkImageCreateInfo imageCreateCI(vks::initializers::imageCreateInfo());
		imageCreateCI.imageType = VK_IMAGE_TYPE_2D;
		// Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
		imageCreateCI.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageCreateCI.extent.width = width;
		imageCreateCI.extent.height = height;
		imageCreateCI.extent.depth = 1;
		imageCreateCI.arrayLayers = 1;
		imageCreateCI.mipLevels = 1;
		imageCreateCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateCI.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateCI.tiling = VK_IMAGE_TILING_LINEAR;
		imageCreateCI.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		// Create the image
		VkImage dstImage;
		VK_CHECK_RESULT(vkCreateImage(device, &imageCreateCI, nullptr, &dstImage));
		// Create memory to back up the image
		VkMemoryRequirements memRequirements;
		VkMemoryAllocateInfo memAllocInfo(vks::initializers::memoryAllocateInfo());
		VkDeviceMemory dstImageMemory;
		vkGetImageMemoryRequirements(device, dstImage, &memRequirements);
		memAllocInfo.allocationSize = memRequirements.size;
		// Memory must be host visible to copy from
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &dstImageMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, dstImage, dstImageMemory, 0));

		// Do the actual blit from the swapchain image to our host visible destination image
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		// Transition destination image to transfer destination layout
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			dstImage,
			0,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// Transition swapchain image from present to transfer source layout
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			srcImage,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// If source and destination support blit we'll blit as this also does automatic format conversion (e.g. from BGR to RGB)
		if (supportsBlit)
		{
			// Define the region to blit (we will blit the whole swapchain image)
			VkOffset3D blitSize;
			blitSize.x = width;
			blitSize.y = height;
			blitSize.z = 1;
			VkImageBlit imageBlitRegion{};
			imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlitRegion.srcSubresource.layerCount = 1;
			imageBlitRegion.srcOffsets[1] = blitSize;
			imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBlitRegion.dstSubresource.layerCount = 1;
			imageBlitRegion.dstOffsets[1] = blitSize;

			// Issue the blit command
			vkCmdBlitImage(
				copyCmd,
				srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageBlitRegion,
				VK_FILTER_NEAREST);
		}
		else
		{
			// Otherwise use image copy (requires us to manually flip components)
			VkImageCopy imageCopyRegion{};
			imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageCopyRegion.srcSubresource.layerCount = 1;
			imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageCopyRegion.dstSubresource.layerCount = 1;
			imageCopyRegion.extent.width = width;
			imageCopyRegion.extent.height = height;
			imageCopyRegion.extent.depth = 1;

			// Issue the copy command
			vkCmdCopyImage(
				copyCmd,
				srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&imageCopyRegion);
		}

		// Transition destination image to general layout, which is the required layout for mapping the image memory later on
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			dstImage,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// Transition back the swap chain image after the blit is done
		vks::tools::insertImageMemoryBarrier(
			copyCmd,
			srcImage,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		vulkanDevice->flushCommandBuffer(copyCmd, queue);

		// Get layout of the image (including row pitch)
		VkImageSubresource subResource { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
		VkSubresourceLayout subResourceLayout;
		vkGetImageSubresourceLayout(device, dstImage, &subResource, &subResourceLayout);

		// Map image memory so we can start copying from it
		const char* data;
		vkMapMemory(device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
		data += subResourceLayout.offset;

		std::ofstream file(filename, std::ios::out | std::ios::binary);

		// ppm header
		file << "P6\n" << width << "\n" << height << "\n" << 255 << "\n";

		// If source is BGR (destination is always RGB) and we can't use blit (which does automatic conversion), we'll have to manually swizzle color components
		bool colorSwizzle = false;
		// Check if source is BGR
		// Note: Not complete, only contains most common and basic BGR surface formats for demonstration purposes
		if (!supportsBlit)
		{
			std::vector<VkFormat> formatsBGR = { VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM };
			colorSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), swapChain.colorFormat) != formatsBGR.end());
		}

		// ppm binary pixel data
		for (uint32_t y = 0; y < height; y++)
		{
			unsigned int *row = (unsigned int*)data;
			for (uint32_t x = 0; x < width; x++)
			{
				if (colorSwizzle)
				{
					file.write((char*)row+2, 1);
					file.write((char*)row+1, 1);
					file.write((char*)row, 1);
				}
				else
				{
					file.write((char*)row, 3);
				}
				row++;
			}
			data += subResourceLayout.rowPitch;
		}
		file.close();

		std::cout << "Screenshot saved to disk" << std::endl;

		// Clean up resources
		vkUnmapMemory(device, dstImageMemory);
		vkFreeMemory(device, dstImageMemory, nullptr);
		vkDestroyImage(device, dstImage, nullptr);

		//screenshotSaved = true;
	}


	void draw()
	{
		VulkanExampleBase::prepareFrame();

		VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSemaphore graphicsWaitSemaphores[] = { compute.semaphore, semaphores.presentComplete };
		VkSemaphore graphicsSignalSemaphores[] = { graphics.semaphore, semaphores.renderComplete };

		// Submit graphics commands
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		submitInfo.waitSemaphoreCount = 2;
		submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
		submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
		submitInfo.signalSemaphoreCount = 2;
		submitInfo.pSignalSemaphores = graphicsSignalSemaphores;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();

		// Wait for rendering finished
		VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		// Submit compute commands
		VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
		computeSubmitInfo.waitSemaphoreCount = 1;
		computeSubmitInfo.pWaitSemaphores = &graphics.semaphore;
		computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
		computeSubmitInfo.signalSemaphoreCount = 1;
		computeSubmitInfo.pSignalSemaphores = &compute.semaphore;
		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		// We will be using the queue family indices to check if graphics and compute queue families differ
		// If that's the case, we need additional barriers for acquiring and releasing resources
		graphics.queueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
		compute.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		loadAssets();
		setupDescriptorPool();
		prepareGraphics();
		prepareCompute();
		buildCommandBuffers();
		prepared = true;
	}

	int framenum = 0;

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		updateComputeUniformBuffers();
		if (camera.updated) {
			updateGraphicsUniformBuffers();
		}
        std::string s = std::to_string(framenum);
		std::string sPadded = "img_000" + std::string(6 - s.length(), '0') + s  + ".ppm";
		framenum++;
		saveScreenshot(sPadded.c_str());
	}
};

VULKAN_EXAMPLE_MAIN()
