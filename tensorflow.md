# Tensorflow 2.x

Автор видео: [selfedu](https://www.youtube.com/@selfedu_rus).  
Ссылка на плейлист [Tensorflow 2.x](https://www.youtube.com/playlist?list=PLA0M1Bcd0w8ynD1umfubKq1OBYRXhXkmH).

+ [Урок 1](#урок-1) - Вычислительные графы
+ [Урок 2](#урок-2) - Переменные, константы
+ [Урок 3](#урок-3) - Операции над переменными
+ [Урок 4](#урок-4) - Работа с GradientTape
+ [Урок 5](#урок-5) - Задачи оптимизации, оптимизаторы
+ [Урок 6](#урок-6) - Полносвязный слой на tf
+ [Урок 7](#урок-7) - Два слоя, tf.Module
+ [Урок 8](#урок-8) - tf.function
+ [Урок 9](#урок-9) - keras.layers.Layer, keras.layers.Model
+ [Урок 10](#урок-10) - keras.Sequential
+ [Урок 11](#урок-11) - Функциональное API, Conv2D на tf, автоенкодер
+ [Урок 12](#урок-12) - Проблемы градиентов, ResNet
+ [Урок 13](#урок-13) - Пример ResNet для CIFAR-10
+ [Урок 14](#урок-14) - Настройки для model.fit(), callbacks
+ [Урок 15](#урок-15) - Настройки для model.compile(), несколько выходов у модели
+ [Урок 16](#урок-16) - Сохранение/загрузка модели

## Урок 1

Вычислительные графы. Нет дублирования вычислений и параллельность. Любую функцию можно с точностью представить в виде такого графа. Так можно находить численное значение производных.

```py
x = tf.Variable([[2.0]])
y = tf.Variable([[-4.0]])
with tf.GradientTape() as tape:
  f = (x+y)**2 + 2*x*y

df = tape.gradient(f, [x,y])
print(df[0], df[1], sep="\n")
```

В `tape` хранятся все промежуточные значения вычислительного графа.

Граф находит производные по формуле $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial t_1} \cdot \frac{\partial t_1}{\partial x} + \frac{\partial f}{\partial t_2} \cdot \frac{\partial t_2}{\partial x}$.

Пример вычислений и что хранится в графе:

```txt
Прямой обход, вычисление промежуточных значений:

x = 2
y = -4
a(x,y) = x+y = -2
b(a) = a**2 = 4
c(x,y) = 2*x*y = -16
f(b,c) = b+c=  -12

Обратный обход, вычисление производных:

df/db = 1+0 = 1
df/dc = 0+1 = 1

dc/dx = 2*y = 2*(-4) = -8
dc/dy = 2*x = 2*2 = 4

db/da = 2*a = 2*(-2) = -4

da/dx = 1+0 = 1
da/dy = 0+1 = 1

df/dx = df/db * db/da * (da/dx + da/dy * dy/dx) + df/dc * (dc/dx + dc/dy * dy/dx) = 1 * -4 * (1+1*0) + 1 * (-8+4*0) = -4-8 = -12

df/dy = df/db * db/da * (da/dx * dx/dy + da/dy) + df/dc * (dc/dx * dx/dy + dc/dy) = 1 * -4 * (1*0+1) + 1 * (-8*0+4) = -4+4 = 0


Можно найти аналитическое решение производных и проверить значения.
f'x = 2(x+y)+2y = 2x+4y -> 2*2+4*(-4) = 4-16 = -12
f'y = 2(x+y)+2x = 4x+2y -> 4*2+2*(-4) = 8-8 = 0
```

## Урок 2

Константы

```py
a = tf.constant(1, shape=(1,1)) # -> [[1]]
b = tf.constant([1,2,3,4])
```

Нельзя использовать смешанные типы данных. Матрица должна быть прямоугольной.

Можно изменить тип.

```py
a2 = tf.cast(a, dtype=tf.float32)
```

Преобразование тензора в np.array

```py
n1 = np.array(a)
n2 = a.numpy()
```

Переменные

```py
v1 = tf.Variable(-4)
v2 = tf.Variable([1,2,3,4], dtype=tf.float32)
v3 = tf.Variable(a)
```

Изменение значений

```py
v1.assign(5)
v2.assign([5,6,7,8]) # Размерности должны совпадать

v1.assign_add(1)
v1.assign_sub(1)

v1.assign_add([1,1,1,1])
```

`tf.Variable` - ссылочный тип, поэтому чтобы сохранить ссылки рекомендуется не пересоздавать переменную, а использовать assign.

```py
v4 = tf.Variable(v1) # Клонирование
```

Узнать размерность. Возвращает кортеж.

```py
print(v2.shape)
```

Индексирование и срезы как в numpy.

```py
print(v2[1:3])

val0=v2[0]
val0.assign(10) # val0 - ссылка и мы изменим v2
print(v2)
print(val0) # Останется старым, потому что он изменил ссылку, а сам является константой
```

Создать новый тензор из конкретных значений

```py
x = tf.constant(range(10)) + 5 # [5,6,7,8, ..., 14]
x2 = tf.gather(x, [0, 4]) # [5, 9]
```

Получить значение из многомерного тензора.

```py
val = v2[(1,2)] # Вторая строка, третий столбец
val = v2[1,2] # То же самое
val = v2[1][2]

vals = v2[:, 1] # Вернёт тензор, состоящий из второго столбца
```

Изменение размерностей. Работает быстро, потому что не пересоздаёт данные, а только новый вид этих данных, сохраняя ссылки.  
Необходимо, чтобы число элементов совпадало (30 = 5*6), иначе ошибка.

```py
a = tf.constant(range(30))
b = tf.reshape(a, [5, 6])
b2 = tf.reshape(a, [5, -1]) # Вычислит автоматически
```

Транспонирование

```py
b_T = tf.transpose(b, perm=[1, 0]) # Какие оси меняем местами. Порядок важен
```

## Урок 3

Создание заполненного тензора

```py
tf.eye(N, M=none) # Везде 0, но на главной диагонали 1
tf.identity(v) # Копирование
tf.ones(shape)
tf.ones_like(v)
tf.zeros(shape)
tf.zeros_like(v)
tf.fill(shape, value)
tf.range(5)

tf.random.normal(shape, mean=0, stddev=1) # Нормальное распределение
tf.random.truncated_normal(shape, mean=0, stddev=1) # Нормальное распределение, но обрезает на 2*sigma
tf.random.uniform(shape, minval=0, maxval=None) # Равномерное распределение
tf.random.shuffle(v)
tf.random.set_seed(5)
```

Операции над тензрами. Можно двумя способами.

```py
tf.add(a, b)
a + b

tf.subtract(a, b)
a - b

tf.divide(a, b) # Поэлементное деление
a / b
a // b

tf.multiply(a, b) # Поэлементное умножение
a * b

a ** 2
```

```py
tf.tensordot(a, b, axes=0) # Внешнее умножение векторов 1x3 * 1*3 = 3x3
tf.tensordot(a, b, axes=1) # Внутреннее умножение векторов 1x3 * 1x3 = 1x1

tf.matmul(a, b) # Матричное умножение 3x3 * 3x3 = 3x3
a @ b
```

```py
tf.reduce_sum(m) # Сумма всех 3x3 -> 1x1
tf.reduce_sum(m, axis=0) # Сумма по столбцам 3x3 -> 1x3
tf.reduce_sum(m, axis=[0, 1])

tf.reduce_mean(m)
tf.reduce_max(m)
tf.reduce_min(m)
tf.reduce_prod(m) # Произведение
tf.reduce_sqrt(a) # Вектор обязан быть вещественным, поэтому нужен tf.cast
tf.square(a)

tf.sin(a)
# ...

tf.keras.activations.relu(a)
# ...

tf.keras.losses.categorical_crossentropy(a)
# ...
```

## Урок 4

Дифференцирование y=x^2 при x=-2

```py
x = tf.Variable(-2.0) # Обязательно вещественное

with tf.GradientTape() as tape:
  y = x**2

df = tape.gradient(y, x)
print(df) # -4
```

y = x*w + b

```py
w = tf.Variable(tf.random.normal(3, 2))
b = tf.Variable(tf.zeros(2, dtype=tf.float32))
x = tf.Variable([-2.0, 1.0, 3.0])

with tf.GradientTape() as tape:
  y = x @ w + b
  loss = tf.reduce_mean(y**2)

df = tape.gradient(loss, [w, b])
print(df[0]) # Множество производных для w (3x2)
print(df[1]) # Множество производных для b (2)
```

Производная зависит только от переменных

```py
x = tf.Variable(0, dtype=tf.float32)
b = tf.constant(1.5)

with tf.GradientTape() as tape:
  f = (x + b)**2 + 2*b

df = tape.gradient(f, [x,b])
print(df[0]) # 3
print(df[1]) # None, потому что константа
```

Можно запретить вычислять производные для переменной

```py
x = tf.Variable(0, trainable=False)
y = tf.Variable(1) + 5 # Становится константой
```

Можно запретить запись в ленту всех промежуточных значений

```py
with tf.GradientType(watch_accessed_variables=False) as tape:
  tape.watch([x, b]) # Можно потом выборочно включать отслеживание
```

```py
with tf.GradientType(watch_accessed_variables=False) as tape:
  tape.watch(x)

  y = 2 * x
  f = y * y # f тоже будет отслеживаться

df_x = tape.gradient(f,[x, y])
```

Такой код не будет работать, потому что после вызова градиента все промежуточные значения пропадают

```py
df_x = tape.gradient(f,x)
df_y = tape.gradient(f,y) # Будет ошибка
```

Можно сделать ленту постоянной, но удалять нужно вручную.

```py
with tf.GradientType(persistent=True) as tape:
  y = 2 * a
  f = y * y

df_x = tape.gradient(f,x)
df_y = tape.gradient(f,y)

del tape
```

Размерность производной такая же, как и переменных, потому что так работает обратный обход графа.

```py
x = tf.Variable(1.0)

with tf.GradientType() as tape:
  y= [2.0, 3.0] * x ** 2

df = tape.gradient(y,x)
print(df) # 10
```

```py
x = tf.Variable([1.0, 2.0])

with tf.GradientType() as tape:
  y= tf.reduce_sum([2.0, 3.0] * x ** 2)

df = tape.gradient(y,x)
print(df) # [4, 10]
```

Можно использовать условия, потому что считаем производную в конкретной точке.

```py
x = tf.Variable(1.0)

with tf.GradientTape() as tape:
  if(x < 2.0):
    y = tf.reduce_sum([2.0, 3.0] * x ** 2)
  else:
    y = x ** 2
```

Все промежуточные зависимости нужно записать в ленту.

```py
x = tf.Variable(1.0)
y = 2*x

with tf.GradientType() as tape:
  z = y ** 2

df = tape.gradient(z,x)
print(df) # None
```

```py
x = tf.Variable(1.0)

with tf.GradientType() as tape:
  y = 2*x
  z = y ** 2

df = tape.gradient(z,x)
print(df) # 12
```

Ошибка

```py
x = tf.Variable(1.0)

for i in range(10):
  with tf.GradientType() as tape:
    y = 2 * x

  df = tape.gradient(y,x)
  
  x = x + 1 # В этот момент x становится константой и пропадает их ленты
  # Правильно будет x.assign_add(1.0)
```

Не работает с другими пакетами

```py
x = tf.Variable(1.0)

with tf.GradientType() as tape:
  y = 2*x + np.square(x)

df = tape.gradient(z,x)
print(df) # None
```

Не нужно использовать целые числа. Это не ошибка, но лента не сможет посчитать производные, выдаст сообщение в консоль и вернёт None.

```py
x = tf.Variable(1)
# x = tf.Variable(1.0)
```

Неявные связи

```py
x = tf.Variable(1.0)
w = tf.Variable(2.0)

with tf.GradientTape() as tape:
  w.assign_add(x) # Лента теряет связь
  # Правильно: w = w + x
  y = w**2

df = tape.gradient(y,x)
print(df) # None
```

## Урок 5

Задачи оптимизации

Найти k, b в kx+b+noise по 1000 точек. Используем СКО.

```py
k = tf.Variable(0.0)
b = tf.Variable(0.0)

f = k*x+b

loss = tf.reduce_mean(tf.square(y - f))
```

Алгоритм градиентного спуска: $k_{n} = k_{n-1} - \eta \cdot \frac{\partial \text{loss}}{\partial k}$.

```py
TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = k_true * x + b_true + noise

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 500
eta = 0.2

for n in range(EPOCHS):
  with tf.GradientTape() as tape:
    f = k * x + b
    loss = tf.reduce_mean(tf.square(y - f))

  dk, db = tape.gradient(loss, [k, b])

  k.assign_sub(eta * dk)
  b.assign_sub(eta * db)

print(k, b, sep="\n")
```

Это задача не про поиск k и b, а про *минимизацию* функции потерь loss.

Эта функция простая, с одним экстремумом, и её можно было просто продифференцировать и решить систему.

$
\begin{cases}
\frac{\partial \text{loss}}{\partial k} = 0 \\
\frac{\partial \text{loss}}{\partial b} = 0
\end{cases}
$

Но зачастую эти функции содержат миллионы параметров с множеством локальных экстремумов.

Градиентный спуск интуитивный и быстрый, но может застрять в локальном минимуме. Все алгоритмы этому подвержены, только в разной степени. Соответственно, есть другие алгоритмы.

Обычный градиент считает все производные $\nabla J(\theta) = -\eta \cdot \nabla_{\theta} \mathcal{L}(\theta; \mathbf{X}, \mathbf{y})$. Вот это $\nabla$ и означает множество всех частных производных. В примере выше находились производные 1000-мерной функции 500 раз.

Есть стохастический градиентный спуск (SGD), который выбирает переменные мини-батчами.

```py
TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = k_true * x + b_true + noise

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 50
eta = 0.02

BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE

for n in range(EPOCHS):
  for n_batch in range(num_steps):
    x_batch = x[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
    y_batch = y[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]


    with tf.GradientTape() as tape:
      f = k * x_batch + b
      loss = tf.reduce_mean(tf.square(y_batch - f))

    dk, db = tape.gradient(loss, [k, b])

    k.assign_sub(eta * dk)
    b.assign_sub(eta * db)

print(k, b, sep="\n")
```

Из-за того, что градиент считается чаще, можно уменьшить число эпох.

В tf есть уже готовое решение.

```py
TOTAL_POINTS = 1000

x = tf.random.uniform(shape=[TOTAL_POINTS], minval=0, maxval=10)
noise = tf.random.normal(shape=[TOTAL_POINTS], stddev=0.2)

k_true = 0.7
b_true = 2.0

y = k_true * x + b_true + noise

k = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 50
eta = 0.02

BATCH_SIZE = 100
num_steps = TOTAL_POINTS // BATCH_SIZE

opt = tf.optimizers.SDG(learning_rate=0.02)

for n in range(EPOCHS):
  for n_batch in range(num_steps):
    x_batch = x[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]
    y_batch = y[n_batch * BATCH_SIZE : (n_batch+1) * BATCH_SIZE]


    with tf.GradientTape() as tape:
      f = k * x_batch + b
      loss = tf.reduce_mean(tf.square(y_batch - f))

    dk, db = tape.gradient(loss, [k, b])

    # k.assign_sub(eta * dk)
    # b.assign_sub(eta * db)
    opt.apply_gradients(zip([dk, db], [k, b])) # Производные, и куда их применить

print(k, b, sep="\n")
```

Разбивку нужно выполнять самостоятельно. SDG также застревает на локальных минимумах. Кто, если не SDG?

+ **Метод моментов (импульсов)** - сохраняем инерцию, по которой двигались, и это позволяет выбираться из неглубоких ям.

```py
opt = tf.optimizers.SDG(learning_rate=0.02, momentum=0.5)
```

+ **Метод Нестерова** - модификация метода моментов. Мы заглядываем, что будет дальше за градиентом.

```py
opt = tf.optimizers.SDG(learning_rate=0.02, momentum=0.5, nesterov=True)
```

Эти два метода смотрят только на градиент, но не обращают внимание на параметры.

+ **Adagrad** - берёт в расчёт то, что некоторые переменные могут сходиться быстрее, чем другие. Улучшает сходимость при разреженных данных. Главный минус - постоянное уменьшение шага обучения (см. формулы). Также сложно подобрать eta.

```py
opt = tf.optimizers.Adagrad(learning_rate=0.2)
```

+ **Adadelta** - модификация Adagrad. Сохраняет не всю историю изменений градиентов, а только её часть, это влияет на шаг. В остальном такой же.

```py
opt = tf.optimizers.Adadelta(learning_rate=4)
```

+ **RMSProp** - модификация Adagrad. *Вместо хранения истории хранения квадратов по каждому параметру, берёт корень из среднего квадрата градиентов по всем параметрам.*

```py
opt = tf.optimizers.RMSProp(learning_rate=0.01)
```

+ **Adam** - модификация Adagrad. Популярен. Использует сглаженные версии среднего и среднеквадратического градиента.

```py
opt = tf.optimizers.Adam(learning_rate=0.1)
```

Какой оптимизатор выбрать? Точного ответа нет, нужны дополнительные исследования. Но лучше пробовать `Adam` -> `Нестерова` -> *`остальные`*.

## Урок 6

Определим задачу параметрической оптимизации как корректное отображение входов X на выходы Y. Этим занимается модель.

В примере ниже модель, которая принимает два числа и должна их сложить. Здесь модель - это два входных нейрона и один выходной.

```py
import tensorflow as tf

class DenseNN(tf.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
            self.fl_init = True
        
        y = x @ self.w + self.b
        return y

model = DenseNN(1)

x_train = tf.random.uniform(minval=0, maxval=10, shape=(100, 2))
y_train = [a + b for a, b in x_train]

loss = lambda x, y: tf.reduce_mean(tf.square(x - y))
opt = tf.optimizers.Adam(learning_rate=0.01)

EPOCHS = 50
for n in range(EPOCHS):
    for x, y in zip(x_train, y_train):
        x = tf.expand_dims(x, axis=0)
        y = tf.constant(y, shape=(1, 1))
        
        with tf.GradientTape() as tape:
            f_loss = loss(y, model(x))
        
        grads = tape.gradient(f_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    print(f_loss.numpy())

print(model.trainable_variables) # ~[1, 1]

print(model(tf.constant([1.0, 2.0]))) # ~3s
```

keras - это надстройка над tf, дающая более удобный API интерфейс. В данном примере всё сделано на ванильном tf. Иногда keras не даёт необходимый функционал.

## Урок 7

Обучение на рукописных цифрах. Многослойная сеть.

```py
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255 # Norm
y_train = y_train / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28]) # Flatten
y_train = tf.reshape(tf.cast(y_train, tf.float32), [-1, 28*28])

y_train = tf.keras.utils.to_categorical(y_train, 10) # OHE

class DenseNN(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
            self.fl_init = True
        
        y = x @ self.w + self.b

        if self.activate == "relu":
          return tf.nn.relu(y)
        if self.activate == "softmax":
          return tf.nn.softmax(y)
        return y

layer_1 = DenseNN(128)
layer_2 = DenseNN(10, activate="softmax")

def model_predict(x):
  y = layer_1(x)
  y = layer_2(y)
  return y

cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
opt = tf.optimizers.Adam(learning_rate=0.001)

BATCH_SIZE = 32
EPOCHS = 10
TOTAL = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

for n in range(EPOCHS):
  loss = 0
  for x_batch, y_batch in train_dataset:
    with tf.GradientTape() as tape:
      f_loss = cross_entropy(y_batch, model_predict(x_batch))
    
    loss += f_loss
    grads = tape.gradient(f_loss, [layer_1.trainable_variables, layer_2.trainable_variables])
    opt.apply_gradients(zip(grads[0], layer_1.trainable_variables))
    opt.apply_gradients(zip(grads[1], layer_2.trainable_variables))
  
  print(loss.numpy())

y = model_predict(x_test)
y2 = tf.argmax(y, axis=1).numpy()

acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
print(acc)

# acc = tf.metrics.Accuracy()
# acc.update_state(y_test, y2)
# print(acc.result().numpy() * 100)
```

Вынесем модель в отдельный класс. Наследуемся от tf.Module, чтобы иметь дополнительные поля.

```py
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255 # Norm
y_train = y_train / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28]) # Flatten
y_train = tf.reshape(tf.cast(y_train, tf.float32), [-1, 28*28])

y_train = tf.keras.utils.to_categorical(y_train, 10) # OHE

class DenseNN(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
            self.fl_init = True
        
        y = x @ self.w + self.b

        if self.activate == "relu":
          return tf.nn.relu(y)
        if self.activate == "softmax":
          return tf.nn.softmax(y)
        return y

class SequenceModule(tf.Module):
  def __init__(self):
    super.__init__()
    self.layer_1 = DenseNN(128)
    self.layer_2 = DenseNN(10, activate="softmax")

  def __call__(self, x):
    y = layer_1(x)
    y = layer_2(y)
    return y

model = SequenceModule()
print(model.submodules)

cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
opt = tf.optimizers.Adam(learning_rate=0.001)

BATCH_SIZE = 32
EPOCHS = 10
TOTAL = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

for n in range(EPOCHS):
  loss = 0
  for x_batch, y_batch in train_dataset:
    with tf.GradientTape() as tape:
      f_loss = cross_entropy(y_batch, model(x_batch))
    
    loss += f_loss
    grads = tape.gradient(f_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
  
  print(loss.numpy())

y = model(x_test)
y2 = tf.argmax(y, axis=1).numpy()

acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
print(acc)
```

## Урок 8

tf.function для ускорения вычислений. Этот декоратор преобразует код функции в граф (не вычислительный, а скорее на уровне декларации).

+ неизменяемые вычисления производятся только один раз, а дальше переиспользуются
+ независимые вычисления разделяются между потоками и устройствами
+ общие вычисления производятся только один раз

Вынесем обучение в tf.function.

```py
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255 # Norm
y_train = y_train / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28]) # Flatten
y_train = tf.reshape(tf.cast(y_train, tf.float32), [-1, 28*28])

y_train = tf.keras.utils.to_categorical(y_train, 10) # OHE

class DenseNN(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
            self.fl_init = True
        
        y = x @ self.w + self.b

        if self.activate == "relu":
          return tf.nn.relu(y)
        if self.activate == "softmax":
          return tf.nn.softmax(y)
        return y

class SequenceModule(tf.Module):
  def __init__(self):
    super.__init__()
    self.layer_1 = DenseNN(128)
    self.layer_2 = DenseNN(10, activate="softmax")

  def __call__(self, x):
    y = layer_1(x)
    y = layer_2(y)
    return y

model = SequenceModule()
print(model.submodules)

cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
opt = tf.optimizers.Adam(learning_rate=0.001)

BATCH_SIZE = 32
EPOCHS = 10
TOTAL = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

@tf.function # Если не добавить, выполнение и скорость никак не изменятся
def train_batch(x_batch, y_batch):
      with tf.GradientTape() as tape:
        f_loss = cross_entropy(y_batch, model(x_batch))
    
    loss += f_loss
    grads = tape.gradient(f_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return f_loss

for n in range(EPOCHS):
  loss = 0
  for x_batch, y_batch in train_dataset:
    loss += train_batch(x_batch, y_batch)
  
  print(loss.numpy())

y = model(x_test)
y2 = tf.argmax(y, axis=1).numpy()

acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
print(acc)
```

Не обязательно так оборачивать методы обучения, можно любую. В граф переводятся если вызывать функцию внутри функции.

```py
def function_tf(x, y):
  s = tf.zeros_like(x, dtype=tf.float32)
  s = s + tf.matmul(x, y)
  for n in range(10):
    s = s + tf.matmul(s, y) * x

  print(s) # /!\
  return s

def timer(fn):
  def wrapper(*args, **kwargs):
    start = time.time()
    fn(*args, **kwargs)
    dt = time.time() - start
    print(f"Время обработки {dt} сек")
  return wrapper

x = tf.ones([1000, 1000], dtype=tf.float32)
y = tf.zeros_like(x, dtype=tf.float32)

function_tf_graph = tf.function(function_tf)

timer(function_tf)(x, y) # 0.12 сек
timer(function_tf_graph)(x, y) # 0.17 сек
```

Граф отработал медленнее, потому что ему нужно проинициализироваться. Но если запускать эти операции в цикле, то граф будет переиспользовать вычисления с прошлых шагов и будет быстрее.  
В цикле по range(10) время выполнения 1.13 и 0.92.

`print(s)` выведется в консоль только один раз, потому что первый раз функция вызывается для построения графа, и это обычный вызов. Но при дальнейших вызовах будет использоваться граф. Граф ленивый и сохраняет только те операции, которые влияют на итог. `print` такой операцией не является.

Советы при использовании tf.function.

+ Тестировать до установления декоратора и после, чтобы убедиться, что результат совпадает.
+ Создавать переменные `tf.Variable` вне функции и передавать их как аргументы. Это касается `tf.keras.layers`, `tf.keras.Model`, `tf.optimizers` и т.д.
+ Не использовать внутри графа глобальные переменные языка Python (кроме `tf.Variable`).
+ Использовать внутри графа только вычисления через tf. С другими пакетами могут быть проблемы.
+ Для максимального ускорения стоит включать в граф чем больше вычислений.

## Урок 9

Выше мы делали всё вручную, и там было много типовых задач. Например, создание полносвязного слоя.

keras - это высокоуровневая обёртка вокруг tf. Он предоставляет много упрощений, которых хватает для широкого круга задач (но не для всех). Если функционала недостаточно (например, GAN), то нужно спускаться на низкие уровни проектирования и писать модели как это было сделано выше. Но keras и tf взаимосовместимы.

Например, за слои отвечает `tf.keras.layers.Layer`.

```py
class DenseLayer(tf.keras.layers.Layer):
  def __init__(self, units):
    super.__init__()
    self.units = units # Число нейронов

  def build(self, input_shape): # Вызывается один раз
    self.w = self.add_weight(shape=(input_shape[-1], units), initializer="random_normal", trainable=True )
    self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

layer_1 = DenseLayer(10)
y = layer_1(tf.constant([[1.0, 2.0, 3.0]]))
```

При вызове слоя `layer_1(v)` ожидается, что мы работает с батчами, поэтому отправляем `[[...]]`. Это матрица размерностью `BATCH_SIZE x count`.

`self.add_weight` предоставляется через наследование.  
Также добавляются `layer.weights`, `layer.trainable_weights`, `layer.non_trainable_weights`.

Сделаем многослойную модель.

```py
class NeuralNetwork(tf.keras.layers.Layer):
  def __init__(self):
    super.__init__()
    self.layer_1 = DenseLayer(128)
    self.layer_2 = DenseLayer(10)

  def call(self, inputs):
    y = self.layer_1(inputs)
    y = tf.nn.relu(y)
    y = self.layer_2(y)
    y = tf.nn.softmax(y)
    return y

model = NeuralNetwork()
y = model(tf.constant([[1.0, 2.0, 3.0]]))
```

Можно заметить, что модель и слой являются расширяют один и тот же класс `tf.keras.layers.Layer`. model - это более сложный слой, и воспринимать его стоит именно так.

Если нам требуется полноценная модель, то существует `tf.keras.Model`.

+ Этот класс имеет метод `fit()` для обучения.
+ Может сохранять и загружать веса. `save()`, `save_weights()`, `load_weights()`.
+ Может оценивать и прогнозировать. `evaluate()`, `predict()`.

```py
class NeuralNetwork(tf.keras.Model):
  def __init__(self):
    super.__init__()
    self.layer_1 = DenseLayer(128)
    self.layer_2 = DenseLayer(10)

  def call(self, inputs):
    y = self.layer_1(inputs)
    y = tf.nn.relu(y)
    y = self.layer_2(y)
    y = tf.nn.softmax(y)
    return y

model = NeuralNetwork()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) # Shortcuts, то же самое, что и раньше

model.fit(x_train, y_train, batch_size=32, epochs=5)

acc = model.evaluate(x_test, y_test)
```

Можно настраивать размер ошибки. Модель будет считать ошибкой величину `categorical_crossentropy + regular`. Например в VAE ошибкой считалась величина `СКО + дивергенция Кульбака-Лейблера`.

```py
class DenseLayer(tf.keras.layers.layer):
  def __init__(self, units):
    ###

  def build(self, input_shape):
    ###

  def call(self, inputs):
    regular = tf.reduce_mean(tf.square(self.w))
    self.add_loss(regular)

    self.add_metric(regular, name="mean square error") # Метрика для логирования

    return tf.matmul(inputs, self.w) + self.b
```

## Урок 10

Продолжаем переходить на высокие уровни от tf к keras.

Модель часто представляет собой последовательное применение слоёв. Для этого есть класс `tf.keras.Sequential`.

```py
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation="relu")
  tf.keras.layers.Dense(10, activation="softmax")
])

print(model.layers[0].weights) # /!\

model.pop() # Удалить последний слой
model.add(tf.keras.layers.Dense(10, activation="softmax")) # добавить слой

model.summary() # Структура модели
```

В данном случае `print(model.layers[0].weights)` выведет пустой массив, потому что до того, как мы пошлём в модель данные, веса не будут проинициализированы, потому что модель не знает размерность входа.

```py
x = tf.random.uniform([1, 20], 0, 1)
y = model(x)

print(model.layers[0].weights) # Ок
```

Размерность входа можно указать явно, и тогда модель проинициализируется сразу. Best practice.

```py
model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(20, ))
  tf.keras.layers.Dense(128, activation="relu")
  tf.keras.layers.Dense(10, activation="softmax")
])

print(model.layers[0].weights) # Ок
print(model.layers) # 2, потому что Input служебный
```

```py
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation="relu", input_shape=(784, ))
  tf.keras.layers.Dense(10, activation="softmax")
])
```

Построим модель, у которой один вход, два слоя и два выхода от каждого слоя.

```py
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation="relu", input_shape=(784, ))
  tf.keras.layers.Dense(10, activation="softmax")
])

model_ex = tf.keras.Model(inputs = model.inputs, outputs = [layer.output for layer in model.layers])

model.fit()
```

`model_ex` будет содержать те же слои и веса, что и `model`, потому что меняется только конфигурация входом и выходов.

Таким образом можно взять готовую модель и обрезать её.

```py
model_ex = tf.keras.Model(inputs = model.inputs, outputs = model.layers[0].output) # Останавливаемся на первом слое
```

Модели можно вкладывать. Здесь слоем является уже обученная модель.

```py
model_ex_2 = tf.keras.Sequential({
  model, # Модель может быть любой
  tf.keras.layers.Dense(12)
})

model.trainable = False # Отключить обучение model как слоя
# model.layers[0].trainable = False # Отключить обучение конкретного слоя

model_ex_2.fit()
```

## Урок 11

Каждый слой является функтором и может быть вызван.

```py
input = Input(shape=(32,32,3))
l = Conv2D(32, 3, activation="relu")(input)
l = MaxPooling2D(2, padding="same")(l)
l = Conv2D(64, 3, activation="relu")(l)
l = MaxPooling2D(2, padding="same")(l)
l = Flatten()(l)
l = Dense(256, activation="relu")(l)
l = Dropout(0.5)(l)
output = Dense(10)(l)

model = tf.keras.Model(inputs = input, outputs = output)
model.summary()
```

Здесь мы связали слои последовательно. В конкретно этом примере можно было использовать `tf.keras.Sequential`, но на практике бывают сложные модели, которые проще описать таким *графом*.

*обучение модели на датасете CIFAR-10*

Как сделать свёрточный слой через tf без keras.

```py
import tensorflow as tf
from tensorflow import keras

class TfConv2D(tf.Module):
    def __init__(self, kernel=(3, 3), channels=1, strides=(2, 2), padding='SAME', activate="relu"):
        super().__init__()
        self.kernel = kernel
        self.channels = channels
        self.strides = strides
        self.padding = padding
        self.activate = activate
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            # Размер матрицы весов [kernel_x, kernel_y, input_channels, output_channels]
            self.w = tf.random.truncated_normal((self.kernel[0], self.kernel[1], x.shape[-1], self.channels), stddev=0.1, dtype=tf.float32)
            self.b = tf.zeros([self.channels], dtype=tf.float32)
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
            self.fl_init = True

        y = tf.nn.conv2d(x, self.w, strides=(1, self.strides[0], self.strides[1], 1), padding=self.padding) + self.b

        if self.activate == "relu":
            return tf.nn.relu(y)
        elif self.activate == "softmax":
            return tf.nn.softmax(y)
        
        return y

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

layer1 = TfConv2D((3, 3), 32) # Слой с весами
y = layer1(tf.expand_dims(x_test[0], axis=0))
print(y.shape) # (1, 16, 16, 32) Изображение уменьшилось до 16x16 и стало иметь 32 канала. Так работает свёртка.

y = tf.nn.MaxPooling(y, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME") # Служебный слой без весов
print(y.shape) # (1, 8, 8, 32)
```

Пример автоенкодера.

```py
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Encoder
enc_input = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu')(enc_input)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Flatten()(x)
enc_output = layers.Dense(8, activation='linear')(x)

encoder = models.Model(enc_input, enc_output, name="encoder")

# Decoder
dec_input = layers.Input(shape=(8,), name="encoded_img")
x = layers.Dense(7 * 7 * 8, activation='relu')(dec_input)
x = layers.Reshape((7, 7, 8))(x)
x = layers.Conv2DTranspose(64, 5, strides=(2, 2), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(32, 5, strides=(2, 2), activation='linear', padding='same')(x)
x = layers.BatchNormalization()(x)
dec_output = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

decoder = models.Model(dec_input, dec_output, name="decoder")

# Autoencoder
autoencoder_input = layers.Input(shape=(28, 28, 1), name="img")
x = encoder(autoencoder_input)
autoencoder_output = decoder(x)

autoencoder = models.Model(autoencoder_input, autoencoder_output, name="autoencoder")

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train, x_train, batch_size=32, epochs=1)

# Predict and visualize the results
# [*1]
h = encoder.predict(tf.expand_dims(x_test[0], axis=0))
img = decoder.predict(h)

plt.subplot(121)
plt.imshow(x_test[0].squeeze(), cmap='gray')
plt.subplot(122)
plt.imshow(img.squeeze(), cmap='gray')
plt.show()
```

В автоенкодере используются ссылки на те же веса, что и в оригинальных моделях. Поэтому когда мы обучили автоенкодер, обучились обе модели, что проверяется в `[*1]`.

## Урок 12

Слои модели VGG-16:  
Conv3-64, Conv3-64, MaxPool,  
Conv3-128, Conv3-128, MaxPool,  
Conv3-256, Conv3-256, Conv3-256, MaxPool,  
Conv3-512, Conv3-512, Conv3-512, MaxPool,  
Conv3-512, Conv3-512, Conv3-512, MaxPool,  
Dense-4096, Dense-4096, Dense-1000, Softmax

Слои модели VGG-19:  
Conv3-64, Conv3-64, MaxPool,  
Conv3-128, Conv3-128, MaxPool,  
Conv3-256, Conv3-256, Conv3-256, **Conv3-256**, MaxPool,  
Conv3-512, Conv3-512, Conv3-512, **Conv3-512**, MaxPool,  
Conv3-512, Conv3-512, Conv3-512, **Conv3-512**, MaxPool,  
Dense-4096, Dense-4096, Dense-1000, Softmax

VGG-16 содержит 16 весовых слоев (13 свёрточных и 3 полносвязных).  
VGG-19 содержит 19 весовых слоев (16 свёрточных и 3 полносвязных).

`Conv3 -> Conv3` даёт эффект свёртки как Conv5. Но при этом имеет только 20 параметров, которые нужно обучить (`(3*3+1)*2`) против 26 (`5*5+1`). Соответственно, применив свёртку три раза подряд охват будет 7x7, а это 30 параметров против 50.

*Согласно экспериментам* использование свёрток подряд улучшает обобщающие способность сети. NVIDIA в библиотеке cuDNN специально оптимизировали именно такой подход.

**Проблема исчезающих (vanishing) градиентов.**  
Проблема обратного распространения ошибки при использовании большого числа полносвязных слоёв. Слои обучаются неравномерно. Согласно формуле градиентного спуска шаг обучения тем больше, чем больше ошибка. В итоге последние слои обучаются и блокируют обучение начальных слоёв.

Есть обратная **проблема взрывающихся (exploding) градиентов**.  
Это проблема рекурсивных нейронных сетей, где постоянно используются одни и те же слои. Небольшой градиент на первых итерациях будет быстро расширяться на последующих итерациях.

Возможное решение - обучать фрагменты модели частями. Это делали с помощью **ограниченной машины Больцмана**. Этот подход неэффективен, но был полезен для поиска решения проблемы.

**Batch Normalization** для ускорения схождения градиентного спуска. Также новые оптимизаторы эту проблему решили.

Теперь можно было бы строить огромные модели, но нет.  
Часто более сложные модели решают задачу хуже простых. Причём ошибка не уменьшается как на обучающей выборке, так и на тестовой. Т.е. это не переобучение, а именно проблема архитектуры на уровне математики.

Microsoft Research предлагают использовать **глубокое остаточное обучение (deep residual learning)**. На этом подходе основан **ResNet**.

Идея в том, чтобы делать блоки слоёв, которые работают как обычно. Но к выходу прибавлять вход.  
Схематично это выглядит вот так:

```txt
input -> [layer_1 -> layer_2 -> layer3] -> [...]
  └---->----------->---------->----------┘
```

Задача аппроксимировать каждый блок, и таким образом градиент не будет затухать на всех блоках.

Есть разные вариации ResNet, где, например, на обходном пути есть дополнительные gates, через которые пропускается сигнал. Это особенно эффективно для RNN. Но и для обычных тоже улучшает обучение.

Для сравнения:

+ **VGG-19** - 19 слоёв
+ **GoogLeNet** - 22 слоя
+ **ResNet** - в первом варианте было 152 слоя, сейчас 1000+ слоёв

## Урок 13

Поскольку это не последовательная модель, будем её определять функционально.

```txt
Input
  ↓
Conv2D
  ↓
Conv2D
  ↓
MaxPooling
  ├→----------------┐
  ↓                 |
[Conv2D -> Conv2D]  |
  ↓                 |
  ↓ ←---------------┘
  +
  ├→----------------┐
  ↓                 |
[Conv2D -> Conv2D]  |
  ↓                 |
  ↓ ←---------------┘
  +
  ↓ 
Conv2D
  ↓
GlobalAveragePooling2D
  ↓
Dense
  ↓
Dropout
  ↓
Dense
```

```py
inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)

block_2_output = layers.add([x, block_1_output]) # Суммирование прямого прохода и обходного

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)

block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x) # Вычисляет среднее число по карте признаков. Возвращает [BATCH_SIZE, channels]
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x) # Рандомно отключает половину нейронов, чтобы избежать переобучения
outputs = layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
```

## Урок 14

validation_split - берёт последние несколько строк обучения до перемешивания и использует их для проверки качества обучения.

```py
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.3)

model.fit(x_train_70, y_train_70, batch_size=64, epochs=5, validation_data=(x_train_30, y_train_30)) # То же самое, но можно разбить выборку вручную своим способом
```

Сформировать выборку можно средствами tf. Можно перевести np.array в тензоры.

```py
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(64)

model.fit(train_dataset, epochs=5, batch_size=100, shuffle=True)
```

`batch_size=100` не будет работать, потому что датасет уже разбит на батчи.
`shuffle=True` не будет работать. Когда мы отправляем Dataset, модель на каждой эпохе обходит массив по порядку.

`steps_per_epoch` - определяет, сколько батчей будет на каждой эпохе. Батчи берутся по очереди и не сбрасываются между эпохами. Если число строк меньше, чем `steps_per_epoch * epochs`, будет WARNING.

Обычно мы предполагаем, что данные сбалансированы и количества строк всех классов сопоставимы. Но если это не так, то можно указать множители для классов `class_weight={0:100.0, 1:1.0, 2:0.5}`.  
Также можно сделать и для отдельных строк `sample_weight`. Например, в MNIST некоторые написания числа 1 похожи на 7 и можно сказать обращать особое внимание на конкретные строки.

```py
history = model.fit()
print(history.history) # Возвращает все метрики для каждой эпохи
```

```py
class DigitsLimit(tf.keras.utils.Sequence):
  def __init__(self, x, y, batch_size, max_len=-1):
    ###
    
  def __len__(self):
    ###
  
  def __getitem__(self, idx):
    ###

  def on_epoch_end(self): # Вызывается после каждой эпохи
    ###

sequence = DigitsLimit(x_train, y_train, batch_size=64, max_len=100) # Берёт только 100 строк
model.fit(sequence, epochs=5, shuffle=True)
```

Функции обратного вызова (callbacks).

`Останавливаем обучение`, если `потери` не изменяются больше `0.5` на протяжении `10` эпох.

```py
model.fit(x, y, epochs=5, callbacks=[
  tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.5, patience=10, verbose=1)
])
```

+ **BaseLogger** - сбор средних значений метрик
+ **History** - запись чего-то в историю во время обучения
+ ***ModelCheckpoint*** - сохранение модели с периодичностью
+ **TerminateOnNaN** - останавливает обучение, если потери стали NaN
+ ...

Можно создать свои функции, если унаследовать класс `tf.keras,callbacks.Callback`.

## Урок 15

У оптимизаторов, функции потерь и метрик есть шорткаты вроде `adam`, но если нужны особые параметры нужно вызывать соответствующую функцию.

Также можно создать свои потери. Нужно использовать только функции из tf, потому что потери потом учитываются в дифференцировании.

```py
def my_loss(y_true, y_pred):
  return tf.reduce_mean(tf.square(y_true - y_pred))

model.compile(..., loss=my_loss)
```

Также можно определять функции потерь через класс, если требуется передать настройки.

$\text{loss} = \sum_i (\alpha \cdot y_{\text{true}}[i] - \beta \cdot y_{\text{pred}}[i])^2
$

```py
class MyLoss(tf.keras.losses.Loss):
  def __init__(self, alpha=1.0, beta=1.0):
    super.__init__()
    self.alpha = alpha
    self.beta = beta

  def call(x_pred, y_pred):
    return tf.reduce_mean(tf.square(self.alpha * y_true - self.beta * y_pred))

model.compile(..., loss=MyLoss(alpha=0.4, beta=0.5))
```

Аналогично можно сделать метрики. Так выглядит дубликат accuracy.

```py
class CategoricalTruePositives(tensorflow.keras.metrics.Metric):
    def __init__(self, name="my_metric"):
        super().__init__(name=name)
        self.true_positives = self.add_weight(name="acc", initializer="zeros")
        self.count = tf.Variable(0.0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        y_true = tf.reshape(y_true, axis=1, shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        
        values = tf.cast(values, "float32")
        self.true_positives.assign_add(tf.reduce_mean(values))
        self.count.assign_add(1.0)

    def result(self):
        return self.true_positives / self.count

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.count.assign(0.0)

```

У модели может быть несколько выходов. В примере будет показан автоенкодер, который возвращает восстановленное изображение и класс. Для этого нужно определить модель функционально.

Также стоит иметь в виду, что для автоенкодера нужно брать потери как СКО, а для классификации кроссентропию.

```py
enc_input = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu')(enc_input)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(64, 4, activation='relu')(x)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(64, 4, activation='relu')(x)
x = layers.MaxPooling2D(2, padding='same')(x)
hidden_output = layers.Dense(8, activation='linear')(x)

x = layers.Dense(7 * 7 * 8, activation='relu')(hidden_output)
x = layers.Reshape((7, 7, 8))(x)
x = layers.Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(64, 2, strides=(2, 2), activation='linear', padding='same')(x)
x = layers.BatchNormalization()(x)
dec_output = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same', name='dec_output')(x)

x2 = layers.Dense(128, activation='relu')(hidden_output)
class_output = layers.Dense(10, activation='softmax', name='class_output')(x2)

model = models.Model(enc_input, [dec_output, class_output])

# Можно задать потери так
model.compile(optimizer=models.optimizers.Adam(learning_rate=0.01),
              loss=['mean_squared_error', 'categorical_crossentropy'],
              metrics=[models.metrics.CategoricalAccuracy()]
              loss_weight=[0.5, 1.4]) # Множители для ошибок

# Или по значению name слоя
model.compile(optimizer=models.optimizers.Adam(learning_rate=0.01),
              loss={'dec_output': 'mean_squared_error',
                    'class_output': 'categorical_crossentropy'},
              metrics={'dec_output': None,
                       'class_output': 'acc'})

# Обучающих выхода тоже два
model.fit(x_train, [x_train, y_train], epochs=1)
```

## Урок 16

Сохранять модели можно ещё и во время обучения на каких-то чекпоинтах, чтобы не терять прогресс, например, при переобучении (говорилось раньше).

```py
model.save("save/path/model")
model = tf.keras.models.load_model("load/path/model")
```

Путь - это папка с таким содержимым:

+ assets
+ **variables** - веса
  + variables.data-{id}
  + variables.index
+ **model.pb** - архитектура модели и конфигурация обучения (состояние оптимизатора, потери и метрики )

По умолчанию модели сохраняются в новом формате `SavedModel`. Старый формат `h5` сохраняет только архитектуру, веса и настройки из compile. Он не сохраняет состояние оптимизатора и продолжить обучение нельзя. Использовать его не рекомендуется.

```py
model.save("model.h5") # Для сохранения в h5 нужно это указать явно. Будет один файл.
```

Сохранение кастомной модели.

```py
class NeuralNetwork(tf.keras.Model):
  def __init__(self, units):
    super.__init__()
    self.units=units
    self.model_layers = [tf.keras.layers.Dense(n, activation="relu" for n in self.units)]

  def call(self, inputs): # Используется при загрузке
    x = inputs
    for layer in self.model_layers:
      x = layer(x)

    return x

model = NeuralNetwork([128, 10])
model.save("path") # Будет WARNING, потому что не реализован compile, но остальное сохранится
```

Если мы загружаем модель, загружаются ещё и методы активации, поэтому если мы хотим использовать кастомный класс, он работать не будет.

```py
class NeuralNetworkLinear(tf.keras.Model):
  def __init__(self, units):
    super.__init__()
    self.units=units
    self.model_layers = [tf.keras.layers.Dense(n, activation="linear" for n in self.units)]

  def call(self, inputs):
    x = inputs
    for layer in self.model_layers:
      x = layer(x)

    return x

model.save("path") # Модель с активацией relu

loaded_model = tf.keras.models.load_model("path", custom_objects={"LinearNetwork": NeuralNetworkLinear}) # Будет всё ещё использовать relu, потому что так сказано в сохранении
```

Чтобы это решить можно переопределить сохранение настроек модели.

```py
class NeuralNetwork(tf.keras.Model):
  def __init__(self, units):
    super.__init__()
      ###

  def call(self, inputs):
    ###

  def get_config(self): # Переопределение родительского метода
    return {units: self.units} # Этого достаточно для восстановления в данном случае

  @classmethod
  def from_config(cls, config):
    return cls(**config) # Вызывает конструктор
```

Можно сохранять и загружать в JSON. Но они сохранят только архитектуру модели, веса и прочее сохраняются как обычно.

```py
model.to_json("path")
model = tf.keras.models.model_from_json("path")
```

Можно извлекать и устанавливать веса, например, если обучается модель частями. То же самое можно делать со слоями. Также моно сохранять и загружать их на диск.

```py
model.predict(x)
model2.predict(x)

# Веса инициализируются только после первого прогона данных через модель (говорилось раньше). Иначе будет ошибка, что весов нет.
weights = model.get_weights()
model2.set_weights(weights)

model.save_weights("path")
model2.load_weights("path") # Или указать .h5
```
