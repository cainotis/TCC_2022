# Atividade 1 - Direção manual

Nesta atividade, apresentaremos o básico do ambiente de simulação de robôs autônomos que usaremos durante a disciplina. 
Vamos usar uma versão modificada do simulador [Duckietown](https://github.com/duckietown/gym-duckietown/), que
chamaremos de [Duckievillage](https://gitlab.uspdigital.usp.br/mac0318-2021/duckievillage). 
O simulador reproduz o comportamento do duckiebot, ou seja, descreve o funcionamento de um robô de direção diferencial com duas rodas controladas por motores de conrrente contínua através da tensão elétrica aplicada em cada motor.
Para familiarizar-se com a estrutura de um agente do simulador, você deverá construir um sistema de controle manual do robô a partir das teclas do seu teclado.

## Pré-requisito

Vamos assumir que você instalou o simulador e clonou o projeto de atividades executando o arquivo [install.sh](../../install.sh) do projeto [Duckievillage](https://gitlab.uspdigital.usp.br/mac0318-2021/duckievillage). Antes de cada atividade, lembre-se de ativar o ambiente do Anaconda:

```bash
conda activate duckietown
```

## Controle manual

O arquivo [manual.py](./manual.py) contém um código mínimo criando um ambiente de simulação e um agente móvel. 
Antes de continuar, preencha o cabeçalho do arquivo com seu nome e número USP (para que possamos identificar corretamente sua submissão no e-disciplinas).

O arquivo importa as seguintes bibliotecas.

```python
import sys
import pyglet
from pyglet.window import key
from duckievillage import create_env
```

A biblioteca [pyglet](http://pyglet.org/) serve para tratar as entradas do teclado, e a temporização da simulação. 
A biblioteca `duckievillage` contém o simulador; vamos usar a função `create_env`, que constrói o
ambiente de simulação que usaremos durante o curso. Esta função retorna uma instanciação do simulador na variável `env`:

```python
env = create_env(
  ...
)
```

O arquivo [manual.py](./manual.py) é comentado com os detalhes sobre o restante das operações necessárias para inciar o simulador. 
Você pode inspecionar o código, mas não será necessário entender o funcionamenteo do simulador para as atividades da disciplina (vamos focar apenas no funcionamento do robô). 
O comportamento do robô é implementado na função `send_commands`, que é chamada periodicamente simulando a latência entre observação e processamento do robô real.
A função recebe um valor `dt` indicando o tempo decorrido desde a última chamada e deve chamar a função `self.env.step` para enviar comandos aos atuadores, isto é, aos motores esquerdo e direito. 
O simulador limite tais valores ao intervalo $`[-1,1]`$, simulando assim limites físicos reais da atuação.

## Tarefa

Modifique o arquivo para implementar um controle remoto manual usando o teclado como entrada. 
Procure valores que permitam ao robô mover-se com velocidade adequada, isto é,  rápido mas de maneira estável.
Quando estiver satisfeito com o resultado, submeta o arquivo `manual.py` no e-discipinas.

