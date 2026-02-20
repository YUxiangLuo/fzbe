// scripts/hash-password.ts

const password = process.argv[2];

// 检查是否提供了密码作为命令行参数
if (!password) {
  console.error("Usage: bun run scripts/hash-password.ts <password-to-hash>");
  process.exit(1);
}

// 使用 Bun 的内置功能来哈希密码
// 默认使用 Argon2id，这是目前推荐的安全算法
const hash = await Bun.password.hash(password, {
  algorithm: "bcrypt", // 使用 bcrypt 以便与 seed.sql 中的格式兼容
  cost: 10, // bcrypt 的计算成本
});

console.log('Generated Hash:');
console.log(hash);
