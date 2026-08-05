import unittest


class McpDependencyContractTest(unittest.TestCase):
    def test_fastmcp_import_used_by_server_is_available(self):
        from mcp.server.fastmcp import FastMCP

        self.assertIsNotNone(FastMCP)


if __name__ == "__main__":
    unittest.main()
