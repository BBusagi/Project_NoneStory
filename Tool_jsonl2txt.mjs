import fs from "fs/promises";
import path from "path";

// 解码 Unicode（\uXXXX）为正常字符
function decodeUnicode(str) {
  return str.replace(/\\u[\dA-Fa-f]{4}/g, (match) => {
    return String.fromCharCode(parseInt(match.replace("\\u", ""), 16));
  });
}

async function main() {
  const inputFile = "output/test8/batch_67e93c387f548190b71e0093fc7687db_output.jsonl";
  const outputDir = "output/test8";

  await fs.mkdir(outputDir, { recursive: true });

  const content = await fs.readFile(inputFile, "utf-8");
  const lines = content.trim().split("\n");

  for (let i = 0; i < lines.length; i++) {
    try {
      const parsed = JSON.parse(lines[i]);
      const raw = parsed.response?.body?.choices?.[0]?.message?.content ?? "(空)";

      // 解码 Unicode（用 JSON 转义 + 解码）
      const unicodeEncoded = raw.replace(/\\u[\dA-Fa-f]{4}/g, (match) => {
        return String.fromCharCode(parseInt(match.replace("\\u", ""), 16));
      });
      
      const filename = path.join(outputDir, `result_${i + 1}.txt`);
      await fs.writeFile(filename, unicodeEncoded, "utf-8");
      
      console.log(`✅ 已保存: ${filename}`);
    } catch (e) {
      console.warn(`❌ 第 ${i + 1} 行处理失败:`, e.message);
    }
  }

  console.log("🎉 全部保存完成！");
}

main();
