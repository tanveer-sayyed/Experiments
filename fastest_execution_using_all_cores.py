from concurrent.futures import ProcessPoolExecutor as ppe

def f(a):
    g = 0
    for i in range(100000000):
        g += 1
    return a.upper()

if __name__ == '__main__':
    results = []
    alphabets = 'a,b,c,d,e,f,g,h,i,j,k,l'.split(',')
    with ppe(max_workers=None) as executor:
        for i, value in zip(range(len(alphabets)),
                            executor.map(f,alphabets)):
            results.append(value)