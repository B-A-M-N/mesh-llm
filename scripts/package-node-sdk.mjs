#!/usr/bin/env node

import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync
} from 'node:fs'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const sourceDir = join(repoRoot, 'sdk', 'node')
const sourceNativeDir = join(sourceDir, 'native')
const expectedTargets = [
  'darwin-arm64',
  'darwin-x64',
  'linux-arm64',
  'linux-x64',
  'win32-x64'
]

const [addonRootArg, outputDirArg] = process.argv.slice(2)
if (!addonRootArg || !outputDirArg) {
  console.error('usage: scripts/package-node-sdk.mjs <addon-root> <output-dir>')
  process.exit(2)
}

const addonRoot = resolve(addonRootArg)
const outputDir = resolve(outputDirArg)
if (existsSync(outputDir) && readdirSync(outputDir).length > 0) {
  throw new Error(`output directory must be empty: ${outputDir}`)
}

const packageJson = JSON.parse(readFileSync(join(sourceDir, 'package.json'), 'utf8'))
validatePackageMetadata(packageJson)

mkdirSync(outputDir, { recursive: true })
cpSync(sourceDir, outputDir, {
  recursive: true,
  filter: (source) => resolve(source) !== sourceNativeDir
})
copyFileSync(join(repoRoot, 'LICENSE'), join(outputDir, 'LICENSE'))

for (const target of expectedTargets) {
  const source = join(addonRoot, target, 'mesh_llm_nodejs.node')
  if (!existsSync(source) || statSync(source).size === 0) {
    throw new Error(`missing non-empty Node SDK addon: ${source}`)
  }
  const destination = join(outputDir, 'native', target, basename(source))
  mkdirSync(dirname(destination), { recursive: true })
  copyFileSync(source, destination)
}

console.log(`prepared ${packageJson.name}@${packageJson.version} in ${outputDir}`)
console.log(`included native addons: ${expectedTargets.join(', ')}`)

function validatePackageMetadata(packageJson) {
  if (packageJson.name !== '@meshllm/sdk') {
    throw new Error(`unexpected Node SDK package name: ${packageJson.name}`)
  }
  if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(packageJson.version)) {
    throw new Error(`invalid Node SDK package version: ${packageJson.version}`)
  }
  if (packageJson.repository?.url !== 'git+https://github.com/Mesh-LLM/mesh-llm.git') {
    throw new Error('Node SDK repository URL must match Mesh-LLM/mesh-llm for npm provenance')
  }
  if (packageJson.repository?.directory !== 'sdk/node') {
    throw new Error('Node SDK repository directory must be sdk/node')
  }
  if (packageJson.publishConfig?.access !== 'public') {
    throw new Error('Node SDK publishConfig.access must be public')
  }
  if (packageJson.publishConfig?.registry !== 'https://registry.npmjs.org/') {
    throw new Error('Node SDK publishConfig.registry must be the public npm registry')
  }
}
