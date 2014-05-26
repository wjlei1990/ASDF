# Visual styles for various elements (nodes) and relations (edges)
# see http://graphviz.org/content/attrs
# Largely copied from
# https://github.com/trungdong/prov/blob/master/prov/model/graph.py
PROV_REC_ENTITY = {"shape": "oval", "style": "filled", "fillcolor": "#FFFC87",
                   "color": "#808080"}
PROV_REC_ACTIVITY = {"shape": "box", "style": "filled", "fillcolor": "#9FB1FC",
                     "color": "#0000FF"}
PROV_REC_AGENT = {"shape": "house", "style": "filled", "fillcolor": "#FED37F"}
PROV_REC_BUNDLE = {"shape": "folder", "style": "filled",
                   "fillcolor": "aliceblue"}
# Relations
PROV_REC_GENERATION = {"label": " wasGeneratedBy", "fontsize": "10.0",
                       "color": "darkgreen", "fontcolor": "darkgreen"}
PROV_REC_USAGE = {"label": " used", "fontsize": "10.0", "color": "red4",
                  "fontcolor": "red"}
PROV_REC_COMMUNICATION = {"label": " wasInformedBy", "fontsize": "10.0"}
PROV_REC_START = {"label": " wasStartedBy", "fontsize": "10.0"}
PROV_REC_END = {"label": " wasEndedBy", "fontsize": "10.0"}
PROV_REC_INVALIDATION = {"label": " wasInvalidatedBy", "fontsize": "10.0"}
PROV_REC_DERIVATION = {"label": " wasDerivedFrom", "fontsize": "10.0"}
PROV_REC_ATTRIBUTION = {"label": " wasAttributedTo", "fontsize": "10.0",
                        "color": "#FED37F"}
PROV_REC_ASSOCIATION = {"label": " wasAssociatedWith", "fontsize": "10.0",
                        "color": "#FED37F"}
PROV_REC_DELEGATION = {"label": " actedOnBehalfOf", "fontsize": "10.0",
                       "color": "#FED37F"}
PROV_REC_INFLUENCE = {"label": " wasInfluencedBy", "fontsize": "10.0",
                      "color": "grey"}
PROV_REC_ALTERNATE = {"label": " alternateOf", "fontsize": "10.0"}
PROV_REC_SPECIALIZATION = {"label": " specializationOf", "fontsize": "10.0"}
PROV_REC_MENTION = {"label": " mentionOf", "fontsize": "10.0"}
PROV_REC_MEMBERSHIP = {"label": " hadMember", "fontsize": "10.0"}

ANNOTATION_STYLE = {'shape': 'note', 'color': 'gray', 'fontcolor': 'black',
                    'fontsize': '10'}
ANNOTATION_LINK_STYLE = {'arrowhead': 'none', 'style': 'dashed',
                         'color': 'gray'}
ANNOTATION_START_ROW = '<<TABLE cellpadding=\"1\" border=\"0\">'
ANNOTATION_ROW_TEMPLATE = """ <TR>
<TD align=\"left\"><FONT color=\"#555555\">%s</FONT></TD>
<TD align=\"left\">%s</TD>
</TR>"""
ANNOTATION_END_ROW = ' </TABLE>>'