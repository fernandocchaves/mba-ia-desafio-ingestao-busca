from search import search_prompt


def print_header():
    """
    Exibe o cabeçalho do chat.
    """
    print("\n" + "="*60)
    print("Chat com IA - Sistema de Busca Semântica em PDF")
    print("="*60)
    print("\nDigite sua pergunta ou 'sair' para encerrar.\n")


def main():
    """
    Função principal que gerencia o loop do chat interativo.
    """
    print_header()
    
    # Inicializa a chain de busca
    try:
        chain = search_prompt()
        
        if not chain:
            print("[ERRO] Não foi possível iniciar o chat. Verifique os erros de inicialização.")
            return
            
    except Exception as e:
        print(f"[ERRO] Erro ao inicializar o chat: {e}")
        print("\nVerifique se:")
        print("  1. O arquivo .env está configurado corretamente")
        print("  2. O banco de dados está rodando (docker compose up -d)")
        print("  3. A ingestão do PDF foi executada (python src/ingest.py)")
        return
    
    # Loop principal do chat
    while True:
        try:
            # Solicita pergunta do usuário
            print("-" * 60)
            pergunta = input("\nFaça sua pergunta: ").strip()
            
            # Verifica se o usuário quer sair
            if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\nEncerrando o chat. Até logo!\n")
                break
            
            # Verifica se a pergunta está vazia
            if not pergunta:
                print("[AVISO] Por favor, digite uma pergunta válida.")
                continue
            
            # Processa a pergunta
            print("\nBuscando informações...")
            resposta = chain.invoke({"pergunta": pergunta})
            
            # Exibe a resposta
            print("\nRESPOSTA:")
            print("-" * 60)
            print(resposta)
            print()
            
        except KeyboardInterrupt:
            print("\n\nChat interrompido pelo usuário. Até logo!\n")
            break
            
        except Exception as e:
            print(f"\n[ERRO] Erro ao processar a pergunta: {e}")
            print("Tente novamente ou digite 'sair' para encerrar.\n")


if __name__ == "__main__":
    main()