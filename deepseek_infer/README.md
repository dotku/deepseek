# deepseek_infer

## Prerequire

```sh
pip install deepspeed transformers accelerate bitsandbytes
```

## Get Started

```sh
deepspeed --hostfile ./hostfile generate.py --deepspeed --deepspeed_config ds_config.json
```
