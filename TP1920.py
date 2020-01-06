import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import math

# 
def resizeImageAndConvert ():
    #cropSize = 255, 255
    #image = Image.open('bruno_normal.JPG')
    #image.thumbnail(cropSize, Image.ANTIALIAS)
    #image.save('test.gif', 'GIF', quality=88)
    
    #img = Image.open('bruno_normal.JPG')
    #new_img = img.resize((255,255))
    #new_img.save("test.gif", "GIF", optimize=True)
    
    im = Image.open('nascimento_oculos_smile.JPG')
    width, height = im.size  
    new_width = 2500
    new_height = 2500
    
    # Setting the points for cropped image  
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Cropped image of above dimension  
    # (It will not change orginal image)  
    new_img = im.crop((left, top, right, bottom)) 
    newsize = (255, 255) 
    new_img = new_img.resize(newsize) 
    # Shows the image in image viewer  
    new_img.save("test1.gif", "GIF", optimize=True, quality=10)

# Funcao de processamento do dataset
def readImages ():
    # Leitura das imagens
    imgs = glob.glob("DatasetMML/*.gif")
    data = [Image.open(i).convert('L') for i in imgs]
    
    # Tamanho do dataset
    size = len(data)
    
    # Passar as imagens para um array
    X = np.array([data[i].getdata() for i in range(size)])
    return X, data, size


# Implementacao do PCA
def pca(X, confidence=0.8):
    # Media do dataset
    mean = np.mean(X,0)
    
    # Centrar os dados
    phi = X - mean
    
    # Calcular os vetores e valores proprios atraves do SVD
    eigenvectors, sigma, variance = np.linalg.svd(phi.transpose(), full_matrices=False)
    eigenvalues = sigma*sigma
    
    # Ordenacao dos valores pp
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # Determinar o n. de vectores pp a usar
    k = 0
    traco = np.sum(eigenvalues)
    while(np.sum(eigenvalues[:k])/traco < confidence):
        k = k+1
    print(k)
    
    # Escolher os vetores pp associados
    eigenvectors = eigenvectors[:,0:k]
    return eigenvalues, eigenvectors, phi, mean, variance


# Calculo dos coeficientes da projeccao
def coefProj(phi, eigenvectors, size):
            
    coef_proj = [np.dot(phi[i], eigenvectors) for i in range(size)]
    #coef_proj = np.reshape(coef_proj, (eigenvectors.shape[1], size))
    return coef_proj


# Verificar se identifica ou nao o input
def testar (input_img , mean, eigenvectors , eigenvalues , size , coef_proj , distance = "mahalanobis"):
    
    # Centrar o input
    gamma = np.array(input_img.getdata())
    test_phi = gamma - mean
    
    # Calcular os coeficientes da projeccao do input
    test_coef_proj = np.dot(test_phi , eigenvectors)
    
    if distance == "euclidian":
        #dist = [np.linalg.norm( coef_proj[i] - test_coef_proj ) for i in range (size)]
        dist = [euclidian(coef_proj[i], test_coef_proj) for i in range (size)]
        
        d_min = round(np.min(dist),2)
        d_max = round(np.max(dist),2)
        limit = 7600
    elif distance == "mahalanobis" :
        dist = mahalanobis(coef_proj , test_coef_proj , eigenvalues , eigenvectors.shape [1])
        d_min = round(np.min(dist),4)
        d_max = round(np.max(dist),4)
        limit = 0.8
    else: 
        print("Distancia invalida .")
        return (-1)
    
    if d_min < limit:
        print('Imagem nr.: '+str(np.argmin(dist))+'\n'+'Distancia minima: '+ str(d_min)+ '\nDistancia máxima: '+ str(d_max)+'\n')
        return dist, test_coef_proj
    else: 
        print('Falhou no reconhecimento.')
        return (-1)


# Distancia euclidiana
def euclidian(x, y):
    if x.size != y.size:
        return (-1)
    z = y - x
    distance = math.sqrt(sum(z**2))
    return distance

# Distance de Mahalanobis
def mahalanobis(x, y, eigenvalues, k):
    if len(x[0]) != len(y):
        return (-1) 
    N = len(x)
    distance =[]
    for i in range(N):
        distance.append(np.sum(np.divide((x[i]-y)**2, eigenvalues[:k]))) 
    return distance


