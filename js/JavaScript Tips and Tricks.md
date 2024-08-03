# JavaScript Tips and Tricks

Автор видео: [As For JS](https://www.youtube.com/@AsForJS)  
Ссылка на плейлист [JavaScript Tips and Tricks](https://www.youtube.com/playlist?list=PL3ziSA8uO7KmGT6pyDRLKheyICRfD3jog)

+ [Идентификаторы](#идентификаторы) - Работа функции с контекстом

## Идентификаторы

В зависимости от того, как будут использоваться переменные, код будет выполняться с разной производительностью.  
Здесь будет два примера.

Если идентификатор используется за пределами внутри которой он определён, это приводит к ситуации, когда интерпретатор должен создать контекст для этой функции.

```js
function doThing() {
    var theThing = 3;

    function doInnerThing() {
        var counter = 0;
        while (theThing--) { // theThing объявлен вне текущей функции, а выше
            counter++;
        }
        return counter;
    }
    doInnerThing();
}
doThing();
```

```assembly
[doThing>]
   31 E> 0000000676B520D6 @    0 : 82 00 01          CreateFunctionContext [0], [1]
         0000000676B520D9 @    3 : 1a f9             PushContext r1
         0000000676B520DB @    5 : 7f 01 00 02       CreateClosure [1], [0], #2
         0000000676B520DF @    9 : c3                Star0
   56 S> 0000000676B520E0 @   10 : 0d 03             LdaSmi [3]
   56 E> 0000000676B520E2 @   12 : 25 02             StaCurrentContextSlot [2]
  224 S> 0000000676B520E4 @   14 : 60 fa 00          CallUndefinedReceiver0 r0, [0]
         0000000676B520E7 @   17 : 0e                LdaUndefined
  241 S> 0000000676B520E8 @   18 : a8                Return

[doInnerThing>]
  115 S> 000000B72B5521DE @    0 : 0c                LdaZero
         000000B72B5521DF @    1 : c3                Star0
  144 S> 000000B72B5521E0 @    2 : 16 02             LdaCurrentContextSlot [2]
         000000B72B5521E2 @    4 : 74 00             ToNumeric [0]
         000000B72B5521E4 @    6 : c2                Star1
         000000B72B5521E5 @    7 : 51 00             Dec [0]
  144 E> 000000B72B5521E7 @    9 : 25 02             StaCurrentContextSlot [2]
         000000B72B5521E9 @   11 : 0b f9             Ldar r1
         000000B72B5521EB @   13 : 96 0a             JumpIfToBooleanFalse [10] (000000B72B5521F5 @ 23)
  163 S> 000000B72B5521ED @   15 : 0b fa             Ldar r0
         000000B72B5521EF @   17 : 50 01             Inc [1]
         000000B72B5521F1 @   19 : c3                Star0
  129 E> 000000B72B5521F2 @   20 : 88 12 00          JumpLoop [18], [0] (000000B72B5521E0 @ 2)
  196 S> 000000B72B5521F5 @   23 : 0b fa             Ldar r0
  211 S> 000000B72B5521F7 @   25 : a8                Return
```

В байт-коде видно, что в функции **doThing** создался контекст `CreateFunctionContext`, создаётся переменная со значением 3 `LdaSmi` и она помещается в контекст `StaCurrentContextSlot`.  
Внутри функции **doInnerThing** происходит загрузка переменной из контекста `LdaCurrentContextSlot`, её изменение и обратная загрузка числа в контекст `StaCurrentContextSlot`.

Эти операции куда более затратные, чем просто получение значения из идентификатора, который объявлен локально. Та же переменная counter функционирует только как регистр r0. Оно создалось через `Star0` и использовалось как `Ldar r0`.  
И если TurboFan заметит этот r0, он преобразует этот байт-код в машинный код и поместит r0 в регистр процессора, в это самая быстрая память, которая существует. В отличии от переменной theThing, которая всегда будет доставаться из контекста.

Второй пример, как можно это исправить.  
Нужно передать переменную как аргумент, чтобы она стала объявленной локально.

```js
function doThing() {
    var theThing = 3;
    function doInnerThing(theThing) {
        var counter = 0;
        while (theThing--) {
            counter++;
        }
        return counter;
    }
    doInnerThing(theThing);
}

doThing();
```

```assembly
[doThing>]
   16 E> 000001ED56B120C6 @    0 : 7f 00 00 02       CreateClosure [0], [0], #2
         000001ED56B120CA @    4 : c2                Star1 
   41 S> 000001ED56B120CB @    5 : 0d 03             LdaSmi [3]
         000001ED56B120CD @    7 : c3                Star0 
  220 S> 000001ED56B120CE @    8 : 61 f9 fa 00       CallUndefinedReceiver1 r1, r0, [0]
         000001ED56B120D2 @   12 : 0e                LdaUndefined 
  245 S> 000001ED56B120D3 @   13 : a8                Return 

[doInnerThing>]
  109 S> 0000004AD0E521BE @    0 : 0c                LdaZero
         0000004AD0E521BF @    1 : c3                Star0
  136 S> 0000004AD0E521C0 @    2 : 0b 03             Ldar a0
         0000004AD0E521C2 @    4 : 74 00             ToNumeric [0]
         0000004AD0E521C4 @    6 : c2                Star1
         0000004AD0E521C5 @    7 : 51 00             Dec [0]
         0000004AD0E521C7 @    9 : 18 03             Star a0
         0000004AD0E521C9 @   11 : 0b f9             Ldar r1
  136 E> 0000004AD0E521CB @   13 : 96 0a             JumpIfToBooleanFalse [10] (0000004AD0E521D5 @ 23)
  155 S> 0000004AD0E521CD @   15 : 0b fa             Ldar r0
         0000004AD0E521CF @   17 : 50 01             Inc [1]
         0000004AD0E521D1 @   19 : c3                Star0
  121 E> 0000004AD0E521D2 @   20 : 88 12 00          JumpLoop [18], [0] (0000004AD0E521C0 @ 2)
  189 S> 0000004AD0E521D5 @   23 : 0b fa             Ldar r0
  207 S> 0000004AD0E521D7 @   25 : a8                Return
```

Теперь обе переменные theThing и counter объявлены как регистры r0 и r1. А далее TurboFan поместит их в регистры уже не виртуальной машины, а процессора.

Где контекст?  
V8 пытается оптимизировать свою работу даже на уровне байт-кода и создаёт контекст только тогда, когда он нужен. Спецификация требует создавать его всегда, но это лишние расходы.

Архитектурно, очень удобно бывает, когда переменные можно создать в родительском окружении. Но даже код, приведённый ниже, будет работать быстрее.

```js
function doThing() {
    var theThing = 3;

    function doInnerThing() {
        var counter = 0;
        var localThing = theThing;

        while (localThing--) {
            counter++;
        }
        
        theThing = localThing;
        
        return counter;
    }
    doInnerThing();
}
doThing();
```

В этом случае контекст всё ещё будет создан, но обратимся мы к нему всего два раза. Остальные операции будут производиться с локальным идентификатором localThing.

Резюмируя.  
Если использовать переменную из родительского окружения, это приводит к двум вещам: создание контекста при каждом вызове doThing и сохранение/загрузка при каждом действии с переменной theThing.