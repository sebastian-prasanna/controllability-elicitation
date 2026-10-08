# GEPA run provider audit (regenerated 2026-09-17 from gepa/provider_audit.json)

reasoning.effort was never set in any GEPA run (gepa.py builds GenerateConfig without extra_body/provider), so every row is at the provider-default effort with free OpenRouter routing.

| sweep | model | eval files | samples | # providers | provider shares |
|---|---|---|---|---|---|
| initial_sweep | deepseek/deepseek-v4-pro-0813 | 4 | 11928 | 14 | DigitalOcean 11%, GMICloud 11%, Alibaba 10%, Sail Research 9%, StreamLake 9%, Novita 8%, Parasail 8%, SiliconFlow 7%, Phala 7%, Together 6%, Cloudflare 6%, Fireworks 5%, BaseTen 2%, DeepInfra 1% |
| initial_sweep | moonshotai/kimi-k3 | 4 | 11928 | 12 | Fireworks 12%, Sail Research 11%, DigitalOcean 10%, Moonshot AI 10%, Chutes 9%, Together 9%, Modal 9%, DeepInfra 9%, Phala 8%, Morph 8%, Parasail 3%, Makora 2% |
| initial_sweep | openai/gpt-oss-120b | 4 | 11928 | 17 | DeepInfra 30%, CoreWeave 14%, DigitalOcean 13%, AkashML 8%, Novita 6%, Mancer 2 6%, Google 5%, Groq 3%, Parasail 3%, BaseTen 2%, Nebius 2%, Amazon Bedrock 2%, SambaNova 2%, Mara 1%, Phala 0%, Cerebras 0%, Together 0% |
| initial_sweep | openai/gpt-oss-20b | 4 | 11928 | 11 | Darkbloom 29%, Parasail 22%, CoreWeave 21%, Novita 12%, Amazon Bedrock 5%, Groq 5%, DeepInfra 4%, Google 1%, Phala 0%, Fireworks 0%, Together 0% |
| initial_sweep | qwen/qwen3-32b | 6 | 17892 | 4 | DeepInfra 41%, SiliconFlow 30%, Nebius 24%, Groq 5% |
| initial_sweep | qwen/qwen3-8b | 4 | 11928 | 1 | Alibaba 100% |
| initial_sweep | z-ai/glm-5.3 | 4 | 11928 | 1 | Z.AI 100% |
| initial_sweep | z-ai/glm-5.3-flash | 4 | 11928 | 15 | Z.AI 52%, GMICloud 15%, Novita 13%, Cloudflare 6%, Together 2%, Io Net 2%, Modal 2%, Parasail 2%, Venice 1%, Reka 1%, DigitalOcean 1%, Wafer 1%, BaseTen 1%, Friendli 0%, DeepInfra 0% |
| new_models_sweep | moonshotai/kimi-k2.6 | 2 | 8928 | 17 | Decart 14%, CoreWeave 11%, Chutes 10%, StreamLake 10%, Inceptron 9%, Parasail 8%, SiliconFlow 6%, Novita 5%, DigitalOcean 4%, Baidu 4%, AtlasCloud 4%, Cloudflare 4%, Moonshot AI 4%, Venice 3%, DeepInfra 3%, GMICloud 1%, Crusoe 1% |
| new_models_sweep | nvidia/nemotron-3-ultra-550b-a55b | 4 | 17856 | 2 | BaseTen 58%, Venice 42% |
| new_models_sweep | qwen/qwen3.8-2.4t-a95b | 2 | 8928 | 7 | Modal 20%, SiliconFlow 17%, Alibaba 17%, Novita 16%, DeepInfra 11%, Venice 11%, Together 7% |
| new_models_sweep | qwen/qwen3.8-27b | 2 | 8928 | 11 | Phala 16%, AkashML 11%, Chutes 11%, Parasail 10%, Alibaba 9%, CoreWeave 8%, Novita 8%, Venice 6%, Io Net 6%, Reka 6%, Cloudflare 6% |
| new_models_sweep | z-ai/glm-5.2 | 2 | 8928 | 27 | DeepInfra 12%, StreamLake 10%, Novita 9%, Sail Research 9%, Ambient 7%, Decart 7%, Inceptron 6%, DigitalOcean 6%, Makora 4%, Alibaba 3%, Mistral 3%, GMICloud 3%, AtlasCloud 2%, SiliconFlow 2%, Fireworks 2%, Phala 2%, Reka 2%, Crusoe 2%, Baidu 2%, BaseTen 2%, Together 2%, Z.AI 1%, Venice 1%, Friendli 1%, Cloudflare 1%, Parasail 1%, CoreWeave 0% |
| second_sweep | deepseek/deepseek-v4-pro-0813 | 8 | 23856 | 13 | Alibaba 16%, StreamLake 14%, GMICloud 12%, Novita 10%, Parasail 9%, SiliconFlow 9%, Phala 8%, DigitalOcean 6%, Cloudflare 5%, Together 4%, DeepInfra 3%, Fireworks 3%, BaseTen 2% |
| second_sweep | moonshotai/kimi-k3 | 8 | 23856 | 12 | Phala 16%, DigitalOcean 13%, Chutes 11%, DeepInfra 10%, Moonshot AI 10%, Fireworks 9%, Parasail 8%, Together 8%, Modal 7%, Morph 3%, Sail Research 3%, Makora 2% |
| second_sweep | openai/gpt-oss-120b | 8 | 23856 | 17 | CoreWeave 25%, DeepInfra 23%, AkashML 13%, Novita 11%, DigitalOcean 7%, Mancer 2 4%, BaseTen 3%, Parasail 3%, Groq 2%, Amazon Bedrock 2%, Together 1%, SambaNova 1%, Nebius 1%, Mara 1%, Phala 1%, Google 1%, Cerebras 0% |
| second_sweep | openai/gpt-oss-20b | 8 | 23856 | 11 | Darkbloom 30%, CoreWeave 18%, Parasail 17%, DeepInfra 12%, Novita 11%, Groq 4%, Amazon Bedrock 4%, Phala 2%, Google 0%, Fireworks 0%, Together 0% |
| second_sweep | qwen/qwen3-32b | 10 | 29820 | 4 | DeepInfra 44%, SiliconFlow 27%, Nebius 25%, Groq 5% |
| second_sweep | qwen/qwen3-8b | 8 | 23856 | 1 | Alibaba 100% |
| second_sweep | z-ai/glm-5.3 | 8 | 23856 | 1 | Z.AI 100% |
| second_sweep | z-ai/glm-5.3-flash | 8 | 23856 | 15 | Z.AI 43%, GMICloud 11%, Together 10%, Modal 9%, Cloudflare 6%, Parasail 5%, Novita 5%, Reka 5%, Io Net 3%, DeepInfra 2%, Wafer 1%, Venice 1%, DigitalOcean 0%, BaseTen 0%, Morph 0% |
| subagent_manual_optimization | moonshotai/kimi-k3 | 40 | 35226 | 15 | Sail Research 9%, Together 8%, Wafer 8%, Morph 8%, Modal 7%, Moonshot AI 7%, Fireworks 7%, Chutes 7%, DigitalOcean 6%, Phala 6%, Makora 6%, Relace 6%, DeepInfra 5%, Parasail 5%, Alibaba 4% |
| subagent_manual_optimization | openai/gpt-oss-20b | 55 | 60168 | 11 | Parasail 24%, Darkbloom 20%, CoreWeave 19%, Novita 8%, Phala 7%, DeepInfra 6%, Amazon Bedrock 4%, Groq 4%, Google 4%, AkashML 2%, Together 0% |
