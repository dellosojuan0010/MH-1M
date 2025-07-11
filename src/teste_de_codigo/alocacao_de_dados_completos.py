from tqdm import tqdm
print("teste de log")
#tqdm(loader, desc=f"Época {epoch+1}/{num_epochs}", unit="batch")
for i in tqdm(range(0,1304,1),"Verificando uma instância"):
    print("avancando")
    for j in range(0,235000): #tqdm(range(0,23500,1),"Verificando atributos"):
        continue