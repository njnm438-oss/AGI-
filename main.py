from agi.agent import AGIAgent
import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['chat','demo','autonomous','test'], default='demo')
    p.add_argument('--llama', type=str, default=None, help='path to ggml llama model')
    args = p.parse_args()

    agent = AGIAgent(llama_model_path=args.llama)
    try:
        if args.mode == 'chat':
            print('Starting chat; type exit to quit')
            while True:
                try:
                    q = input('You: ')
                except (EOFError, KeyboardInterrupt):
                    break
                if q.strip().lower() in ('exit','quit'):
                    break
                print('AGI:', agent.ask(q))
        elif args.mode == 'autonomous':
            for i in range(10):
                emb = agent.perceive_text(f'observation {i}')
                a = agent.autonomous_step(emb)
                print('action', a)
        else:
            agent.perceive_text('Bonjour, je suis AGI Lab v4 prototype', importance=0.7)
            print(agent.ask('Présente-toi en une phrase'))
    finally:
        agent.shutdown()
