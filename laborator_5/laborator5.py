from sys import ps2
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# Defining network structure

game_model = BayesianNetwork(
    [
        ("1st_player", "2nd_player"),
        ("1st_player", "P1"),
        ("2nd_player", "P2"),
        ("P1", "P2"),
        ("1st_player", "P3"),
        ("P1", "P3"),
        ("P2", "P3")
    ]
)

CPD_1st_player = TabularCPD(
    variable='1st_player', 
    variable_card=5, 
    values=[[0.2], [0.2], [0.2], [0.2], [0.2]]
    )

print('1st_player: ', CPD_1st_player)

CPD_2nd_player = TabularCPD(
    variable='2nd_player', 
    variable_card=5, 
    values=[[0, 0.25, 0.25, 0.25, 0.25],                                    
            [0.25, 0, 0.25, 0.25, 0.25], 
            [0.25, 0.25, 0, 0.25, 0.25], 
            [0.25, 0.25, 0.25, 0, 0.25], 
            [0.25, 0.25, 0.25, 0.25, 0]], 
    evidence=['Card_first_player'], 
    evidence_card=[5]
)
                                                            
print('2nd_player: ',CPD_2nd_player)
CPD_P1 = TabularCPD(
    variable='P1', 
    variable_card=2, 
    values=[[0, 0.2, 0.4, 0.8, 1], 
            [1, 0.8, 0.6, 0.2, 0]], 
    evidence=['1st_player'], 
    evidence_card=[5]
)

CPD_P2 = TabularCPD(
    variable='P2', 
    variable_card=3, 
    values=[[0, 0.2, 0.3, 0.4, 1, 0, 0, 0, 0, 0], 
            [1, 0.8, 0.7, 0.6, 0, 1, 0.7, 0.4, 0.3, 0], 
            [0, 0, 0, 0, 0, 0, 0.3, 0.6, 0.7, 1]], 
    evidence=['P1', '2nd_player'], 
    evidence_card=[2, 5]
)


CPD_P3 = TabularCPD(
    variable='P3', 
    variable_card=3, 
    values=[[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
            [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]], 
    evidence=['Card_first_player','P1', 'P2'], 
    evidence_card=[5, 2, 3]
    )

game_model.add_cpds(CPD_1st_player, CPD_2nd_player, CPD_P1, CPD_P2, CPD_P3)

game_model.get_cpds()

checking_model = game_model.check_model()

model_inferential = VariableElimination(game_model)

first_draw = model_inferential.query(
    variables=['P1'],
    evidence={'1st_player': 1}
)
print('first draw:' ,first_draw)

second_draw = model_inferential.query(
    variables=['P2'], 
    evidence={'P1': 0, '2nd_player': 2}
)
print('second draw:', second_draw)

P1_v2 = TabularCPD(
    variable = 'P1', 
    variable_card = 2,
    values =[[1, 0.2, 0.6, 0.3, 0],
             [0, 0.8, 0.4, 0.7, 1]],
    evidence = ['1st_player'],
    evidence_card=[5]
)

P2_v2 = TabularCPD(
    variable = 'P2', 
    variable_card=3, 
    values=[[1, 0.8, 0.4, 0.3, 0, 1, 0.8, 0.4, 0.2, 0],
            [0, 0.2, 0.6, 0.7, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.2, 0.6, 0.8, 1]], 
    evidence=['P1', '2nd_player'], 
    evidence_card=[2, 5]
)

print(P1_v2, P2_v2)


first_draw_v2 = model_inferential.query(
    variables=['P1'], 
    evidence={'1st_player': 1}
)
print('second draw:' , first_draw_v2)

second_draw_v2 = model_inferential.query(
    variables=['P2'], 
    evidence={'P1': 0, '2nd_player': 2}
)
print('second draw:',second_draw_v2)