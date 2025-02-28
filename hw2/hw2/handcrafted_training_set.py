import matplotlib.pyplot as plt

### 2.2 Handcrafted Training Set

## 2 points
#X = [[1,1],[2,1]]
#y = [0,1]

## 4 points
X = [[1,1],[1,2],[2,1],[2,2]]
y = [0,1,1,0]

plt.scatter([x[0] for x in X], [x[1] for x in X], c=y, cmap='bwr', s=100)
plt.xlabel('Feature 0 (x1)')
plt.ylabel('Feature 1 (x2)')
plt.title('Handcrafted Training Set')
plt.xlim(0, 5)
plt.ylim(0, 3)
plt.grid()
plt.show()
