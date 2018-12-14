from namegen import NameGenerator

namegen = NameGenerator()
name, attention = namegen.get_name_and_attention_for('self assert : <num> + <num> equals : <num> .')
print('{', 'name: \'{}\', attention: {}'.format(name, attention), '}')
