from pathlib import Path
from .nlp import FactoryNLP
from .vectors import FactoryVDB
from .services.chunk import DataProcess
from .services import ImageTranslate

vdb = FactoryVDB().create_vdb(provider="chromadb")
vdb.connect("database")

embed = FactoryNLP().create_mtf(provider="huggingface")
embed.connect()
embed.set_model_name()

nlp = FactoryNLP().create_mtf(provider="gemini")
nlp.connect()
nlp.set_model_name("gemini-3-flash-preview")

# service = DataProcess(nlp, vdb)
# service.upload_batches(
#     "phrases",
#     Path(r"C:\Users\ramyu\OneDrive\Desktop\MachineT\assets\data\phrases.xlsx"),
#     "arabic_sentence",
# )

# chunks = [
#     "الفصــل الأول إدارة المستخدمين",
#     "- رقم المجموعة: يظهر رقم مجموعة المستخدمين في هذا الحقل آلياً مع إمكانية التعديل.",
# ]
# nearest = vdb.semantic_search("phrases", nlp.embed(chunks), 3)
# print(nearest)

service = ImageTranslate(vdb, nlp, embed)
results = service.translate_pdf(
    Path(r"C:\Users\ramyu\OneDrive\Desktop\MachineT\assets\data\mini-test.pdf")
)
print(results)
