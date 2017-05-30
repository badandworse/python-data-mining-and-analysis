#%%


obj="""
{
    "names":"Wes",
    "places_lived":["United States","Spain","Germany"],
    "pet":null,
    "siblings":[{"name":"Scott","age":25,"pet":"Zuko"},
                {"name":"Katie","age":33,"pet":"Cisco"}   ]
}
"""
import json

result=json.loads(json_str)
result

